"""Repeated gene-expression perturbations and plateau analysis for human lung cancer."""
from pathlib import Path
import argparse,gc,json,sys
ROOT=Path(__file__).resolve().parent
EXPERIMENT_DIR=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parents[2]/'src'))
from stvirtual.perturb import prepare_runtime
prepare_runtime(EXPERIMENT_DIR)
from stvirtual.perturb import load_config,run
from stvirtual.perturb import preflight
from stvirtual.perturb import calc_alpha_gt_threshold_by_t,find_first_positive_and_plateau_by_piecewise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--runs',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=2026)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=EXPERIMENT_DIR/'artifacts/perturb/perturbation_test')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.runs<=0:parser.error('--runs must be positive')
    if args.dry_run:
        print(json.dumps(preflight(args.config),indent=2));return
    import anndata as ad,numpy as np,pandas as pd,torch
    config,experiment,paths=load_config(args.config)
    readiness=preflight(args.config)
    if not readiness['ready']:
        raise FileNotFoundError('Complete dataset training first. Missing: '+', '.join(readiness['missing_inputs'])+'; model: '+readiness['model_import'])
    if config['mode']!='gene_knockdown':parser.error('LUAD batch requires gene_knockdown config')
    base=args.output.resolve()
    if not base.is_relative_to(EXPERIMENT_DIR):parser.error('Output must be under experiments')
    # Each condition also has independent non-overwrite checks in run().
    base.mkdir(parents=True,exist_ok=False)
    rows=[];diags=[];summaries=[];gene_tables=[]
    for seed in range(args.seed_start,args.seed_start+args.runs):
        c=dict(config);c['seed']=seed
        p=dict(paths);p['output']=str(base/f'seed{seed}')
        try:
            out=run(c,experiment,p,device=args.device)
            report=json.loads((out/'report.json').read_text())
            genes=report['perturbation']['genes']
            tables=[]
            frames=sorted((out/'simulation').glob('*.h5ad'))
            for i,path in enumerate(frames):
                frame=ad.read_h5ad(path)
                obs=frame.obs[['diff_alpha','is_diff']].copy();obs['t']=i;tables.append(obs)
            obs=pd.concat(tables,ignore_index=True)
            history=ad.AnnData(obs=obs)
            prop=calc_alpha_gt_threshold_by_t(history,t_col='t',alpha_col='diff_alpha',is_diff_col='is_diff',only_diff=False,t_min=0,t_max=len(frames)-1,threshold=0.6)
            first,plateau,diag,summary=find_first_positive_and_plateau_by_piecewise(prop,t_col='t_step',prop_col='prop_alpha_gt_0.6',positive_eps=1e-12,smooth=3,min_plateau_len=10)
            diag.insert(0,'seed',seed);summary.insert(0,'seed',seed)
            summary['first_positive_t']=first;summary['plateau_t']=plateau;summary['perturb_num']=len(genes)
            gt=pd.DataFrame({'seed':seed,'gene':genes})
            gp=base/f'gene_list_seed{seed}.csv';dp=base/f'diag_piece_seed{seed}.csv';sp=base/f'alpha_threshold_summary_seed{seed}.csv'
            gt.to_csv(gp,index=False);diag.to_csv(dp,index=False);summary.to_csv(sp,index=False)
            gene_tables.append(gt);diags.append(diag);summaries.append(summary)
            rows.append({'seed':seed,'perturb_num':len(genes),'first_positive_t':first,'plateau_t':plateau,'frame_count':len(frames),'gene_list_path':str(gp),'diag_piece_path':str(dp),'summary_path':str(sp),'genes':'|'.join(genes),'status':'ok'})
        except Exception as error:
            rows.append({'seed':seed,'status':f'failed: {error!r}'})
        finally:
            pd.DataFrame(rows).to_csv(base/'all_runs_overview.csv',index=False)
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        print(f'{seed}: {rows[-1]["status"]}',flush=True)
    for name,tables in [('all_gene_lists',gene_tables),('all_diag_piece',diags),('all_alpha_threshold_summary',summaries)]:
        if tables:pd.concat(tables,ignore_index=True).to_csv(base/f'{name}.csv',index=False)
    if any(row['status']!='ok' for row in rows):raise SystemExit(1)
    print(base)

if __name__=='__main__':main()
