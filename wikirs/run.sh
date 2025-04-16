# train model with WINCEL loss, SkyCLIP pretrained model : 
python train_multi_text.py --criterion wincel --model SkyCLIP   --debug

# train model with InfoNCE loss, SkyCLIP pretrained model and random Wikipedia sentences :
python train_single_text.py --criterion infoNCE --model CLIP --text random   --debug

# train model with top-1 text loss 
python train_multi_text.py --criterion sample_top_1_text_Loss --model SkyCLIP  --debug

# train model with top-p text loss 
python train_multi_text.py --criterion sample_top_p_text_Loss --model SkyCLIP  --debug

# train model with hard bootstraping :
python train_multi_text.py --criterion HardBootstrap --bootstrap_beta 0.9 --model GeoRSCLIP   --debug

# train model with soft bootstraping :
python train_multi_text.py --criterion SoftBootstrap --bootstrap_beta 0.9 --model RemoteCLIP    --debug

# train model with sentences sub-parts :
python train_multi_text.py --criterion wincel --use_sentence_parts  --debug --model GeoRSCLIP


