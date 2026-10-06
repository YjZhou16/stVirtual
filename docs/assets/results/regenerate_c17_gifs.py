"""Convert the 12 existing C17 videos into the eight wiki GIFs; sources stay read-only."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess
JOBS={
 'gp1':['gp1/GP1_normal_cancer.mp4'],
 'luad':['luad/LUAD_AAH_LUAD.mp4'],
 'mbrain':['mouse/Mbrain_T168_T171.mp4'],
 'membryo':['mosta/MOSTA_E9.5_E11.5_slow6s_total7s.mp4','mosta/MOSTA_E11.5_E13.5_slow6s_total7s.mp4','mosta/MOSTA_E13.5_E15.5_slow6s_total7s.mp4'],
 'hmln':['hmln/HMLN_S1_S3.mp4','hmln/HMLN_S4_S6.mp4','hmln/HMLN_S7_S9.mp4'],
 'mcardiac':['heart/Heart_E9.5_E11.5_xyz_timeline.mp4'],
 'imc':['imc/IMC_0_14.mp4'],
 'heart_slide_seq':['heart_slide_seq/HeartSlide_E10.5_E12.5.mp4'],
}
def probe(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(path)]))['streams'][0]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 repo=Path(__file__).resolve().parents[3]
 parser.add_argument('--source-root',type=Path,default=repo.parent/'stVirtual_revision/R2/C17')
 parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
 parser.add_argument('--width',type=int,default=960)
 parser.add_argument('--fps',type=int,default=10)
 parser.add_argument('--workers',type=int,default=2)
 parser.add_argument('--overwrite',action='store_true')
 parser.add_argument('--jobs',nargs='+',choices=list(JOBS),default=list(JOBS))
 args=parser.parse_args()
 if args.width<=0 or args.fps<=0 or args.workers<=0:parser.error('width, fps and workers must be positive')
 source=args.source_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 def convert(job):
  from PIL import Image
  paths=[source/p for p in JOBS[job]]
  for path in paths:
   if not path.is_file():raise FileNotFoundError(path)
  final=out/({'gp1':'human_gastric_cancer.gif','luad':'human_lung_cancer.gif','heart_slide_seq':'Mcardiac_EA.gif','imc':'human_breast_cancer.gif'}.get(job,f'{job}.gif'))
  if final.exists() and not args.overwrite:raise FileExistsError(final)
  partial=out/f'.{job}.partial.gif'
  inputs=['-i',str(paths[0])]
  if len(paths)>1:
   listing=out/f'.{job}.concat.txt'
   listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in paths))
   inputs=['-f','concat','-safe','0','-i',str(listing)]
  filt=f'fps={args.fps},scale={args.width}:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff:max_colors=256[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle'
  cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2',*inputs,'-filter_complex_threads','1','-filter_complex',filt,'-loop','0',str(partial)]
  subprocess.run(cmd,check=True)
  stream=[probe(p) for p in paths];expected=sum(float(s['duration']) for s in stream)
  image=Image.open(partial);duration=0
  for i in range(image.n_frames):image.seek(i);image.load();duration+=image.info.get('duration',0)
  if image.size[0]!=args.width or abs(duration/1000-expected)>1/args.fps+0.02:raise ValueError(f'GIF validation failed: {job}')
  frames=image.n_frames;size=list(image.size);loop=image.info.get('loop');image.close()
  if loop!=0:raise ValueError(f'GIF must loop continuously: {job}')
  os.replace(partial,final)
  if len(paths)>1:listing.unlink()
  result={'gif':final.name,'width':size[0],'height':size[1],'fps':args.fps,'frames':frames,'duration_seconds':duration/1000,'bytes':final.stat().st_size,'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'sources':[{'path':str(p.relative_to(source)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**s} for p,s in zip(paths,stream)]}
  print(f'{job}: {frames} frames, {duration/1000:.2f}s, {final.stat().st_size/1048576:.1f} MiB',flush=True)
  return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(convert,args.jobs))
 (out/'c17_gifs_manifest.json').write_text(json.dumps({'source_root':str(source),'width':args.width,'fps':args.fps,'loop':0,'dither':'bayer_scale=4','jobs':results},indent=2)+'\n')
if __name__=='__main__':main()
