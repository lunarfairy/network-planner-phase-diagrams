"""Select the compact SI presentation while preserving all generated data tables."""
import re

KEEP = ['grids', 'theory', 'mech', 'flow', 'place', 'decomp', 'reward', 'oos']

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
        'Sample flow by condition. Bot, node-rounds controlled by replacement bots; human, recorded human decisions; capital, decisions excluded because the capital constraint forced defection; analysed, retained human decisions.',
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
        r'Placement of empirical parameter vectors under the final-round objective. Class agrees between samples A and B in every row. Both select the same $s^\star$, so sample-B gains also evaluate the locked sample-A choice. Gains are labelled by sample.',
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
    # Display wording and rounded parameter labels; numerical table cells are unchanged.
    assert 'Parameter grids. The rule count varies with the horizon as given in Supplementary Note~1. Unless varied, parameters take the published values ($\\kap=1$ except where varied, $w=0$, $H=15$, $n=16$, donation game). $\\kap$ grids use 16 values in $[0,1.5]$. A=B, percentage of points at which samples A and B give the same class; uncertain, points flagged as in Methods; classes from pooled A+B estimates (E/S/C/P; mixed probation never occurs as simplest class).' in blocks['grids']
    blocks['grids'] = blocks['grids'].replace('Parameter grids. The rule count varies with the horizon as given in Supplementary Note~1. Unless varied, parameters take the published values ($\\kap=1$ except where varied, $w=0$, $H=15$, $n=16$, donation game). $\\kap$ grids use 16 values in $[0,1.5]$. A=B, percentage of points at which samples A and B give the same class; uncertain, points flagged as in Methods; classes from pooled A+B estimates (E/S/C/P; mixed probation never occurs as simplest class).', 'Parameter grids. The rule count varies with the horizon as given in Supplementary Note~1. Unless varied, parameters take the values specified in Methods ($\\kap=1$ except where varied, $w=0$, $H=15$, $n=16$, donation game). $\\kap$ grids use 16 values in $[0,1.5]$. A=B, percentage of points at which samples A and B give the same class; uncertain, points flagged as in Methods; classes from pooled A+B estimates (E/S/C/P; mixed probation never occurs as simplest class).')
    assert 'published, $\\kap=0.6$ & ' in blocks['mech']
    blocks['mech'] = blocks['mech'].replace('published, $\\kap=0.6$ & ', '$\\kap=0.6$ & ')
    assert 'published, $\\kap=0.8$ & ' in blocks['mech']
    blocks['mech'] = blocks['mech'].replace('published, $\\kap=0.8$ & ', '$\\kap=0.8$ & ')
    assert 'published, $\\kap=0.8$, no capital limit & ' in blocks['mech']
    blocks['mech'] = blocks['mech'].replace('published, $\\kap=0.8$, no capital limit & ', '$\\kap=0.8$, no capital limit & ')
    assert 'published, $\\kap=1.0$ & ' in blocks['mech']
    blocks['mech'] = blocks['mech'].replace('published, $\\kap=1.0$ & ', '$\\kap=1.0$ & ')
    assert 'Parameter-replacement decomposition between the published model and the re-estimated M1w model. One block of parameters, or one later-round response coefficient, is exchanged at a time ($\\beta_0$ intercept, $\\beta_s$ degree, $\\beta_n$ number and $\\beta_r$ share of cooperating neighbours). Class on samples A/B; gains averaged over both samples; 10,000 games per sample.' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Parameter-replacement decomposition between the published model and the re-estimated M1w model. One block of parameters, or one later-round response coefficient, is exchanged at a time ($\\beta_0$ intercept, $\\beta_s$ degree, $\\beta_n$ number and $\\beta_r$ share of cooperating neighbours). Class on samples A/B; gains averaged over both samples; 10,000 games per sample.', 'Parameter-replacement decomposition between the parameter settings and pooled M1w estimates in Table~1 of the main text. One block of parameters, or one later-round response coefficient, is exchanged at a time ($\\beta_0$ intercept, $\\beta_s$ degree, $\\beta_n$ number and $\\beta_r$ share of cooperating neighbours). Rows 2--12 start from the parameter settings; rows 13--23 start from the pooled estimates. Arrows give the replacement direction; displayed values are rounded as in Table~1. Class on samples A/B; gains averaged over both samples; 10,000 games per sample.')
    assert 'Published simulator & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published simulator & ', 'Table 1 settings & ')
    assert 'Published, re-estimated $\\kap$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\kap$ & ', '$\\kap$: $1\\to 0.891$ & ')
    assert 'Published, re-estimated $\\beta_0,\\beta_s,\\beta_n,\\beta_r$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_0,\\beta_s,\\beta_n,\\beta_r$ & ', '$\\beta_0,\\beta_s,\\beta_n,\\beta_r$: $(-0.010,-0.193,0.370,1.521)\\to (0.192,-0.255,0.194,4.000)$ & ')
    assert 'Published, re-estimated $\\sigma_\\theta$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\sigma_\\theta$ & ', '$\\sigma_\\theta$: $2.41\\to 3.663$ & ')
    assert 'Published, re-estimated round-1 coefficients & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated round-1 coefficients & ', 'round-1 coefficients: $(1.807,0.818)\\to (1.628,0.670)$ & ')
    assert 'Published, re-estimated acceptance & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated acceptance & ', 'acceptance: $(0.774,0.085,0.287,0.909)\\to (0.756,0.096,0.296,0.885)$ & ')
    assert 'Published, re-estimated $w$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $w$ & ', '$w$: $0\\to 0$ & ')
    assert 'Published, re-estimated $\\beta_0$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_0$ & ', '$\\beta_0$: $-0.010\\to 0.192$ & ')
    assert 'Published, re-estimated $\\beta_s$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_s$ & ', '$\\beta_s$: $-0.193\\to -0.255$ & ')
    assert 'Published, re-estimated $\\beta_n$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_n$ & ', '$\\beta_n$: $0.370\\to 0.194$ & ')
    assert 'Published, re-estimated $\\beta_r$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_r$ & ', '$\\beta_r$: $1.521\\to 4.000$ & ')
    assert 'Published, re-estimated $\\beta_0,\\beta_s,\\beta_n$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Published, re-estimated $\\beta_0,\\beta_s,\\beta_n$ & ', '$\\beta_0,\\beta_s,\\beta_n$: $(-0.010,-0.193,0.370)\\to (0.192,-0.255,0.194)$ & ')
    assert 'Re-estimated, published $\\kap$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\kap$ & ', '$\\kap$: $0.891\\to 1$ & ')
    assert 'Re-estimated, published $\\beta_0,\\beta_s,\\beta_n,\\beta_r$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_0,\\beta_s,\\beta_n,\\beta_r$ & ', '$\\beta_0,\\beta_s,\\beta_n,\\beta_r$: $(0.192,-0.255,0.194,4.000)\\to (-0.010,-0.193,0.370,1.521)$ & ')
    assert 'Re-estimated, published $\\sigma_\\theta$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\sigma_\\theta$ & ', '$\\sigma_\\theta$: $3.663\\to 2.41$ & ')
    assert 'Re-estimated, published round-1 coefficients & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published round-1 coefficients & ', 'round-1 coefficients: $(1.628,0.670)\\to (1.807,0.818)$ & ')
    assert 'Re-estimated, published acceptance & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published acceptance & ', 'acceptance: $(0.756,0.096,0.296,0.885)\\to (0.774,0.085,0.287,0.909)$ & ')
    assert 'Re-estimated, published $w$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $w$ & ', '$w$: $0\\to 0$ & ')
    assert 'Re-estimated, published $\\beta_0$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_0$ & ', '$\\beta_0$: $0.192\\to -0.010$ & ')
    assert 'Re-estimated, published $\\beta_s$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_s$ & ', '$\\beta_s$: $-0.255\\to -0.193$ & ')
    assert 'Re-estimated, published $\\beta_n$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_n$ & ', '$\\beta_n$: $0.194\\to 0.370$ & ')
    assert 'Re-estimated, published $\\beta_r$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_r$ & ', '$\\beta_r$: $4.000\\to 1.521$ & ')
    assert 'Re-estimated, published $\\beta_0,\\beta_s,\\beta_n$ & ' in blocks['decomp']
    blocks['decomp'] = blocks['decomp'].replace('Re-estimated, published $\\beta_0,\\beta_s,\\beta_n$ & ', '$\\beta_0,\\beta_s,\\beta_n$: $(0.192,-0.255,0.194)\\to (-0.010,-0.193,0.370)$ & ')
    assert 'Prediction in all seven human conditions. Observed group means with standard errors; group counts in parentheses. Published and LOCO entries use 10,000 games. LOCO models are estimated without the predicted condition; M3 intervals use 40 group-bootstrap refits per fold (2,000 games each), combined with resampling of observed groups. The 95\\% interval for the improvement in MAE of LOCO M3 over the published simulator is 0.032--0.117. $^a$Simulated with the published encouragement table. Pooled and condition-specific in-sample predictions remain in the Supplementary Data.' in blocks['oos']
    blocks['oos'] = blocks['oos'].replace('Prediction in all seven human conditions. Observed group means with standard errors; group counts in parentheses. Published and LOCO entries use 10,000 games. LOCO models are estimated without the predicted condition; M3 intervals use 40 group-bootstrap refits per fold (2,000 games each), combined with resampling of observed groups. The 95\\% interval for the improvement in MAE of LOCO M3 over the published simulator is 0.032--0.117. $^a$Simulated with the published encouragement table. Pooled and condition-specific in-sample predictions remain in the Supplementary Data.', 'Prediction in all seven human conditions. Observed group means with standard errors; group counts in parentheses. All simulated entries use 10,000 games. LOCO models are estimated without the predicted condition; M3 intervals use 40 group-bootstrap refits per fold (2,000 games each), combined with resampling of observed groups. $^a$Simulated with the encouragement table. Pooled and condition-specific in-sample predictions remain in the Supplementary Data.')
    assert 'Condition & Observed & Published & LOCO M1 & LOCO M3' in blocks['oos']
    blocks['oos'] = blocks['oos'].replace('Condition & Observed & Published & LOCO M1 & LOCO M3', 'Condition & Observed & Table 1 settings & LOCO M1 & LOCO M3')
    assert 'Matched branches: effect of one extra conciliation step at planner steps $s\\le10$ (means over $s$; 20,000 paired games per step; s.e. of the final-round effect about 0.0007). Immediate, round $s+1$; split into the contributions of round-$s$ defectors and cooperators; cumulative, summed over rounds $s+1,\\dots,15$; final, round 15; $\\Delta$ degree, final mean degree.' in blocks['mech']
    blocks['mech'] = blocks['mech'].replace('Matched branches: effect of one extra conciliation step at planner steps $s\\le10$ (means over $s$; 20,000 paired games per step; s.e. of the final-round effect about 0.0007). Immediate, round $s+1$; split into the contributions of round-$s$ defectors and cooperators; cumulative, summed over rounds $s+1,\\dots,15$; final, round 15; $\\Delta$ degree, final mean degree.', 'Matched branches: effect of one extra conciliation step at planner steps $s\\le10$ (means over $s$; 20,000 paired games per step; s.e. of the final-round effect about 0.0007). Immediate, round $s+1$; split into the contributions of round-$s$ defectors and cooperators; cumulative, summed over rounds $s+1,\\dots,15$; final, round 15; $\\Delta$ degree, final mean degree. In the first four rows, unspecified parameters use the settings in Table~1 of the main text; the remaining rows use the pooled M1w estimates.')
    return '\n\n'.join(blocks[k] for k in KEEP)+'\n'
