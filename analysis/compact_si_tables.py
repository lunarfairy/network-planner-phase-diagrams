"""Select the compact SI presentation while preserving all generated data tables."""
import re

KEEP = ['grids', 'theory', 'mech', 'flow', 'quad', 'place', 'decomp', 'reward', 'oos']

def _table(label, caption, header, rows, spec):
    return '\n'.join([r'\begin{table}[htbp]\centering\small',
        r'\caption{'+caption+r'}\label{stab:'+label+'}',
        r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{'+spec+r'}\toprule',
        header+r'\\\midrule', *[r+r'\\' for r in rows],
        r'\bottomrule\end{tabular}\end{adjustbox}\end{table}'])

def _rows(block):
    body=block.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    return [x.strip() for x in body.split(r'\\') if x.strip()]

def compact_si_tables(full):
    blocks={re.search(r'\\label\{stab:([^}]+)\}',b).group(1):b
            for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',full,re.S)}
    assert len(blocks)==16
    blocks['grids']=blocks['grids'].replace('Parameter grids of the certified sweep.',
        'Parameter grids. The rule count varies with the horizon as given in Supplementary Note~1.')
    flow=[]
    for row in _rows(blocks['flow']):
        c=[x.strip() for x in row.split('&')]
        flow.append(' & '.join(c[i] for i in (0,1,2,4,5,6,7)))
    blocks['flow']=_table('flow',
        'Sample flow by condition. Bot, node-rounds controlled by replacement bots; human, recorded human decisions; capital, decisions excluded because the capital constraint forced defection; analysed, retained human decisions. Full node-round counts, coding sensitivities and outcomes over all 16 nodes are in '+r'\texttt{data/processed/sample\_flow.csv}.',
        'Condition & Groups & Players & Bot & Human & Capital & Analysed',flow,'lrrrrrr')
    quad=[]
    for row in _rows(blocks['quad']):
        if row.startswith(('Revised panel, M1 fitted with 160','Revised panel, M1 fitted with 320','Revised panel, M3 fitted with 160')):
            c=[x.strip() for x in row.split('&')]
            name=c[0].replace('Revised panel, ','').replace(' fitted with ','; fit at ').replace(' nodes',' nodes')
            quad.append(' & '.join([name,c[3],c[4],c[6]]))
    assert len(quad)==3
    refits=[]
    for row in _rows(blocks['quad']):
        match=re.search(r'Revised panel, refit (160|320) nodes \(M1\): NLL ([\d.]+),.*?([\d.]+),.*?([\d.]+),.*?([\d.]+) &',row)
        if match:refits.append(' & '.join(match.groups())+r'\\')
    assert len(refits)==2
    blocks['quad']=_table('quad',
        'Quadrature convergence for the current data panel. Upper block, negative log-likelihood (NLL) at fixed parameters evaluated with different numbers of Gauss--Hermite nodes. Lower block, M1 refitted at 160 and 320 nodes. All fits used in the analyses have 160 nodes. Earlier-panel and 24-node comparisons remain in the Supplementary Data.',
        r'Parameters & $Q=96$ & $Q=160$ & $Q=320$',quad,'lrrr').replace(
        r'\bottomrule\end{tabular}\end{adjustbox}\end{table}',
        r'\bottomrule\end{tabular}\end{adjustbox}'+'\n'+r'\par\smallskip'+
        '\n'+r'\begin{tabular}{rrrrr}\toprule'+ '\n'+
        r'Refit nodes & NLL & $\hat\kap$ & $\hat\sigma_\theta$ & $\hat\beta_r$\\\midrule'+'\n'+
        '\n'.join(refits)+'\n'+
        r'\bottomrule\end{tabular}\end{table}')
    place=[]
    for row in _rows(blocks['place']):
        c=[x.strip() for x in row.split('&')]
        cls=c[1] if c[1]==c[2] else c[1]+'/'+c[2]
        place.append(' & '.join([c[0],cls,c[3],c[5],c[6]]))
    blocks['place']=_table('place',
        r'Placement of empirical parameter vectors under the final-round objective. Class agrees between samples A and B in every row. Both select the same $s^\star$, so sample-B gains also evaluate the locked sample-A choice. Gains are labelled by sample. Class S is selected in all 200 bootstrap draws of M1w and all 200 of M3; $s^\star=11$ in 196 and 198 draws, respectively.',
        r'Parameter vector & Class A/B & $s^\star$ & $\Delta_E$ (B) & $\Delta_C$ (A)',place,'llrrr')
    rewardcap=(r'Alternative objectives on the main plane (208 points) and at the re-estimated population. The sample-A argmax is evaluated on B. Winner C counts conciliation-throughout winners; loss is final-round cooperation lost relative to the final-round winner. The last two columns give the winning rule $(s,\ell)$ and loss. Discounted objectives use $\gamma=0.99$ and a rejection weight $P=0$ or 1. The penalty counts rejected recommendations per possible pair at each planner step, with the same discount. Rejection is recorded for the single shown endpoint; the original training reward in ref.~19 counts rejection by both endpoints. These four alternatives accompany the primary final-round objective.')
    blocks['reward']=re.sub(r'\\caption\{.*?\}\\label',lambda m:r'\caption{'+rewardcap+r'}\label',blocks['reward'],count=1,flags=re.S)
    oos=[]
    for row in _rows(blocks['oos']):
        c=[x.strip() for x in row.split('&')]
        if c[0] in ('MAE without GraphNet',r'Spearman $\rho$'):continue
        oos.append(' & '.join(c[i] for i in (0,1,2,5,6)))
    blocks['oos']=_table('oos',
        r'Prediction in all seven human conditions. Observed group means with standard errors; group counts in parentheses. Published and LOCO entries use 10,000 games. LOCO models are estimated without the predicted condition; M3 intervals use 40 group-bootstrap refits per fold (2,000 games each), combined with resampling of observed groups. The 95\% interval for the improvement in MAE of LOCO M3 over the published simulator is 0.032--0.117. $^a$Simulated with the published encouragement table. Pooled and condition-specific in-sample predictions remain in the Supplementary Data.',
        'Condition & Observed & Published & LOCO M1 & LOCO M3',oos,'lllll')
    return '\n\n'.join(blocks[k] for k in KEEP)+'\n'
