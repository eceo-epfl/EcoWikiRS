import time
import os
import wandb
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler
####################################################################
SEED = 24536
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED) 
random.seed(SEED) 
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###################################################################
from utils.argparser import parse_args
from utils.utils import save_args_to_file, get_model
import utils.variables as variables
from utils.variables import NUM_SENTENCES_PER_SAMPLES
from utils.dataset import WikiRSDataset
from utils.losses import *
from run_model_evaluation import run_model_evaluation
from run_eval_zero_shot import eval_zero_shot_eunis

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

def train_model(train_loader,  model, tokenizer, criterion, optimizer, store_feat=False):
    """
    Training loop
    """
    model.train()
    global device     
    tick = time.time()          
    losses=[]
    if store_feat :
        train_tensor,train_eunis = [],[]

  
    for  input in train_loader:
        optimizer.zero_grad()
        with  torch.autocast(device_type=device, dtype=torch.float16,enabled=True):            

            # Get image features : 
            B  = input['image'].shape[0]
            image_features = model.encode_image(input['image'].to(device))
            image_features = image_features/ image_features.norm(dim=-1, keepdim=True)
            
            # Get text features :
            text_tokens = [  tokenizer (x.rsplit('<sep>') ) for x in input['text'] ] 
            # Transform to tensor and reshape :
            text_tokens = pad_sequence(text_tokens, batch_first=True).reshape (B*NUM_SENTENCES_PER_SAMPLES,-1).to(device)
            text_features= model.encode_text(text_tokens)                      
            text_features = text_features/ text_features.norm(dim=-1, keepdim=True)

            # Mask text features that corresponding to padding token (empty strings):
            text_features = text_features.reshape(B,NUM_SENTENCES_PER_SAMPLES,-1) 
            mask =1.- input['text_padding'].unsqueeze(-1).to(device)
            text_features = text_features * mask

            # Compute loss
            loss = criterion( img_feat = image_features,text_feat=text_features)

            if store_feat :
                train_tensor.append(image_features.detach().cpu())
                train_eunis.append(input['eunis'])

        losses.append(loss.item())       
        scaler.scale(loss).backward() #loss.backward()
        scaler.step(optimizer) #    optimizer.step()
        scaler.update()  # Updates the scale for next iteration.            

    results = {
            'train_loss':np.round(np.mean(losses),3),            
            'train_duration':np.round(time.time()-tick,3),
            'lr':optimizer.param_groups[-1]['lr']   ,    
        }

    if store_feat :
        results ['train_tensor'] = torch.cat(train_tensor ,dim=0)
        results ['train_eunis'] = torch.cat(train_eunis,dim=0)
        
    return results 
    
def validate_model(val_loader,  model, tokenizer, criterion, store_feat=False):
    """
    Validation 
    """
    model.eval()
    global device     
    tick = time.time()          
    losses=[]
    alpha_max=[]
    if store_feat :
        val_tensor, val_eunis = [],[]   

    for  input in val_loader:
        with  torch.autocast(device_type=device, dtype=torch.float16, enabled=True), torch.no_grad():

            # Get image features : 
            B  = input['image'].shape[0]
            image_features = model.encode_image(input['image'].to(device))
            image_features = image_features/ image_features.norm(dim=-1, keepdim=True)
            
            # Get text features :
            text_tokens = [  tokenizer (x.rsplit('<sep>') ) for x in input['text'] ] 
            # Transform to tensor and reshape :
            text_tokens = pad_sequence(text_tokens, batch_first=True).reshape (B*NUM_SENTENCES_PER_SAMPLES,-1).to(device)
            text_features= model.encode_text(text_tokens)                      
            text_features = text_features/ text_features.norm(dim=-1, keepdim=True)

            # Mask text features that corresponding to padding token (empty strings):
            text_features = text_features.reshape(B,NUM_SENTENCES_PER_SAMPLES,-1) 
            mask =1.- input['text_padding'].unsqueeze(-1).to(device)
            text_features = text_features * mask

            # Compute loss
            loss = criterion( img_feat = image_features, text_feat=text_features,)
            losses.append(loss.item())

            if store_feat :
                val_tensor.append(image_features.detach().cpu())
                val_eunis.append(input['eunis'])
            
            
            # Compute text image coeff
            a_n = torch.einsum('bij,bjk->bik',text_features,  image_features.unsqueeze(-1)) 
            y = torch.softmax(a_n/criterion.temperature,dim=1)  
            alpha_max += [torch.max(y,dim=0)[0].mean().item()]
            
    results = {
                'val_loss': np.round(np.mean(losses),3),            
                'val_duration': np.round(time.time()-tick,3),   
                'alpha_max' : np.mean(alpha_max),            
            }

    if store_feat :
        results ['val_tensor'] = torch.cat(val_tensor ,dim=0)
        results ['val_eunis'] = torch.cat(val_eunis,dim=0)

    return results 




if __name__ =="__main__":  

    args = parse_args()  


    args.folder = variables.ROOT + 'output/'+args.name
    if not os.path.exists(args.folder):
        os.mkdir(args.folder)
        print('Create exp directory :', args.folder)
    save_args_to_file(args, filename=f"{args.folder}/args.txt")

    if args.log_wandb :
        wandb.init(project = 'WikiRS_release',  
        entity = "zvalerie",
        reinit = True,
        config = vars(args),           
        )    
        wandb.run.name = args.name

    #### Set up training loop :
    model, tokenizer  = get_model(args.model)
    model.to(device);
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                    lr = args.lr, 
                                    weight_decay=args.weight_decay
                                    ) 
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer, 
                                step_size = 2, 
                                gamma = 0.95, 
                                )       
    print('STEP Scheduler by 5pct at each 2 epoch')                       
    
    if args.criterion in [ 'Weighted_InfoNCELoss', 'wincel']:
        criterion = Weighted_InfoNCE_Loss(temperature=args.temperature, device=device, )
    
    elif args.criterion == 'sample_top_1_text_Loss':
        criterion = sample_top_1_text_Loss(temperature=args.temperature, device=device, )

    elif args.criterion == 'sample_top_p_text_Loss':
        criterion = sample_top_p_text_Loss(temperature=args.temperature, device=device, )

    elif args.criterion == 'HardBootstrap':
        criterion = HardBootstrap(temperature=args.temperature, device = device, bootstrap_beta=args.bootstrap_beta)
    
    elif args.criterion == 'SoftBootstrap':
        criterion = SoftBootstrap(temperature=args.temperature, device = device, bootstrap_beta=args.bootstrap_beta)
    
    elif args.criterion == 'infonce':
        raise ValueError(args.criterion +'Should be used with the train_single_text.py file !! ')
        
    else :
        raise ValueError(args.criterion +'is not valid criterion.')
        
    
    dataset_dict = {
                'debug' : args.debug,            
                'article_key' : args.text,
                'use_sentence_parts' :args.use_sentence_parts,
                'one_sentence_per_item' : False,
            }

    # Get dataloaders :         
    train_ds = WikiRSDataset(phase = 'train',**dataset_dict)
    val_ds = WikiRSDataset(phase= 'val',**dataset_dict)
    train_loader = DataLoader(dataset=train_ds,
                            batch_size=args.bs,
                            num_workers=args.num_workers,
                            shuffle=True,
                            drop_last =True,
                            )
    val_loader   = DataLoader(dataset=val_ds,
                            batch_size=args.bs,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=True,
                            )


    # Start model training :     
    best_metric = 0.
    print('-'*50)
    print('Start model training')
    print(args)
    print('-'*50)

    epoch =0
    for epoch in range( args.epoch):
        
        train_metrics = train_model(train_loader, model, tokenizer, criterion, optimizer,store_feat=False)
        val_metrics = validate_model(val_loader, model, tokenizer, criterion,store_feat=False)
        eunis_metrics = eval_zero_shot_eunis(model=model,tokenizer=tokenizer,args=args)

        scheduler.step()            
        print(  f"Epoch {epoch}/{args.epoch}",
                f"train loss: {train_metrics ['train_loss']:.3f}",
                f"val_loss: {val_metrics ['val_loss']:.3f}",
                f"EUNIS metrics F1 {eunis_metrics['f1']:.3f} / OA : {eunis_metrics['overall_acc']:.3f}",
                f"\talpha_max : {val_metrics ['alpha_max']:.3f}",
                )
        if np.isnan(train_metrics ['train_loss']) :
            raise ValueError( f'Train loss is NaN ')   
        
        if eunis_metrics['f1'] > best_metric :
            best_metric = eunis_metrics['f1']

            # Save best model :
            model_path = f"{args.folder}/{args.model}_ft_best.pt"
            checkpoint_dict = {
                "epoch": epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }   
            torch.save(checkpoint_dict,model_path)
            print( f"Model saved: {model_path}" )
        model_path = f"{args.folder}/{args.model}_ft_last.pt"
        checkpoint_dict = {
            "epoch": epoch,
            "name": args.name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }   
        torch.save(checkpoint_dict,model_path)
        print( f"Model saved: {model_path}" )

        if args.log_wandb :
            wandb.log({'val_loss': val_metrics['val_loss'], 'alpha_max': val_metrics['alpha_max']}, step=epoch)
            wandb.log(train_metrics, step=epoch) 
            wandb.log({'eunis_f1':eunis_metrics['f1']}, step=epoch)
            wandb.log({'eunis_oa':eunis_metrics['overall_acc']}, step=epoch)
          
    
     
    print('-'*50)
    print('End of model training')
    print('Saving best model....')
    


    # Save final model :
    model_path = f"{args.folder}/{args.model}_ft_final.pt"
    checkpoint_dict = {
        "epoch": epoch,
        "name": args.name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }   
    torch.save(checkpoint_dict,model_path)
    print( f"Model saved: {model_path}" )    


    
    print('-'*50)
    print('Launch model evaluation')
    print('-'*50)
    run_model_evaluation (args.folder, args.model, f"{args.model}_ft_final")
    run_model_evaluation (args.folder, args.model, f"{args.model}_ft_best")
    

    print('-'*50,'END of training','-'*50)