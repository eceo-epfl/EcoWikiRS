
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from utils.utils import get_model
from utils import variables as v
from utils.dataset import WikiRSDataset
from utils.prompts_template import aerial_dict,rs_dict,sat_dict
from utils.eunis import  EUNIS_id_to_cls, EUNIS_id_to_cls_id, EUNIS_id_to_description, EUNIS_id_to_short_labels 
from utils.utils import read_config_file
from utils.eunis import cls_id_to_EUNIS_id
from utils.utils import read_config_file,get_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_visual_features(folder, base_model, checkpoint_name=None,phase='test'):
    """
    Generates and saves visual features for the test dataset using the specified model weights.
    Args:
        folder (str): Path to the experiment folder where outputs will be saved.
        base_model (str): Name of the base model to use for feature extraction [CLIP, RemoteCLIP, GeoRSCLIP, SkyCLIP].
        checkpoint_name (str, optional): Checkpoint file for fine-tuned model weights.
        phase (str, optional): dataset split to use for feature extraction [train, val, test].
    """
    # Create output folder
    ######################
    checkpoint_path = f"{folder}/{checkpoint_name}.pt"
    out_folder = folder + f'/visual_features/{checkpoint_name}'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if os.path.exists (out_folder+ f"/{phase}_features.pt") :
        print('\t-->Visual features already extracted for ',out_folder.rsplit('output/')[-1],'\n\t-->Move to model evaluation...')
        return

    # Load model 
    ############
    model, _ = get_model(base_model)
    model.eval()  ;
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device);

    
    if checkpoint_name is not None :  
        checkpoint = torch.load(checkpoint_path , weights_only= True,map_location=device);
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)       ; 
        print(msg,'\nFine tuned model loaded from checkpoint ', checkpoint_path )
        model.eval();
    else :
        print(f'No checkpoint name provided, using pretrained {base_model} model weights')
    
    # Get dataset and dataloader
    ############################
    dataset_dict = { 'debug' : False, }           
    test_ds = WikiRSDataset(phase = phase,**dataset_dict)
    dl_dict = {"batch_size":256,"num_workers":16,"shuffle":False}                       
    test_loader = DataLoader(dataset = test_ds, **dl_dict)

    ## Store visual features
    ########################
    with torch.no_grad(),torch.autocast(device_type=device):

        for name,dl in zip([phase],[test_loader]):    
            
            store_tensor,store_eunis,store_id = [],[],[]    

            for  input in tqdm(dl) :
                image_features = model.encode_image(input['image'].to(device))
                image_features = image_features/ image_features.norm(dim=-1, keepdim=True)
                store_tensor.append(image_features.detach().cpu())
                store_eunis.append(input['eunis'])
                store_id.append(input['id'])

            store_tensor = torch.cat(store_tensor,dim=0)
            store_eunis = torch.cat(store_eunis,dim=0)  
            
            fn = f'{out_folder}/{name}_features.pt'
            torch.save({'tensor':store_tensor,'eunis':store_eunis, 'id':store_id},fn)
            print('Visual features saved in ',fn)    
    print('feature generation done for ',out_folder)   
    del model, store_tensor , store_eunis  , store_id 


def eval_prompts(model, tokenizer, prompt_dict, visual_features_path):
    """
    Evaluate overall accuracy and F1-scorefor the given prompt dictionnary.
    Uses stored visual features from the generate_visual_features() function.
    Args:
        model (torch.nn.Module): The model used for encoding text.
        tokenizer (torch.nn.Module): The tokenizer used to preprocess the prompts.
        prompt_dict (dict): A dictionary containing prompts as values for each eunis class.
        visual_features_path (str): Path to the test dataset file (PyTorch tensor file).
    Returns:
        dict: A dictionary containing the evaluation metrics:
            - 'OA' (float): Overall accuracy, rounded to 3 decimal places.
            - 'F1' (float): Macro F1-score, rounded to 3 decimal places.
    """
    test_data = torch.load(visual_features_path,weights_only=False, map_location = device)
    test_tensor,test_labels   = test_data['tensor'].cuda(),test_data['eunis'].cpu().numpy()

    # Load the eunis prompts from prompt_dict and map them to 0-25 :
    prompts = [k for k in list(prompt_dict.values())]
    # Move from eunis id to 0-N indices
    labels = [EUNIS_id_to_cls_id[x] for x in test_labels] 


    if isinstance(prompts[0],list):
        # Encode eunis prompts with encoder :
        text_tokens =   [ tokenizer ( x) .to(device) for x in prompts]
        text_features = [model.encode_text(x) for x in text_tokens]
        text_features = [ (x/x.norm(dim=-1, keepdim=True)).mean(dim=0) for x in text_features]
        text_features = torch.stack([ x/x.norm(dim=-1, keepdim=True) for x in text_features])

    else :
        # Encode eunis prompts with CLIP  text encoder : 
        text_tokens =   tokenizer ( prompts) .to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features/ text_features.norm(dim=-1, keepdim=True)

    assert len(prompts ) == 25

    # Get prediction : 
    similarity = test_tensor @ text_features.T 
    prediction = torch.argmax(similarity,axis=1).detach().cpu().numpy()

    # Get metrics
    overall_accuracy = accuracy_score(y_true = labels, y_pred = prediction)
    _,_,f1,_  = precision_recall_fscore_support(y_true = labels, 
                                        y_pred = prediction, 
                                        average = 'macro',  
                                        zero_division = 0,                                                        
                                        )

    result = {'OA': np.round(overall_accuracy,3),'F1':np.round(f1,3) }            

    return result 



def run_model_evaluation (folder, base_model, checkpoint_name=None):
    """
        Evaluates a model's performance using visual features and various prompt dictionaries.
        Args:
            folder (str): The directory containing the model checkpoints and visual features.
            base_model (str): The name of the base model to be used for evaluation.
            checkpoint_name (str, optional): The name of the checkpoint file (without extension) 
                to load fine-tuned model weights. If None, the base model's pretrained weights are used.
        Returns:
            None: The function saves the evaluation results to a CSV file and prints relevant messages.
        Example:
            run_model_evaluation(
                folder="/path/to/checkpoints",
                base_model="CLIP",
                checkpoint_name="model_checkpoint"
            )
    """

    
    # Look for the visual features:
    ################################  
    checkpoint_path = f"{folder}/{checkpoint_name}.pt"
    outname = checkpoint_path.replace('.pt','_prompt_eval.csv')
    if os.path.exists (outname) :
        print('\t-->Evaluation metrics already computed : ',outname)
        return
    out_folder = folder + f'/visual_features/{checkpoint_name}'
    visual_features_path= out_folder+ "/test_features.pt"
    
    if not os.path.exists (visual_features_path) :
        print('No visual features found in ',visual_features_path)
        return
       # generate_visual_features(folder, base_model, checkpoint_name)
    
    # Load model 
    ############
    model, tokenizer = get_model(base_model)
    model.eval()  ;
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device);

    
    if checkpoint_name is not None :  
        checkpoint = torch.load(checkpoint_path , weights_only= True,map_location=device);
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)       ; 
        print(msg,'\nFine tuned model loaded from checkpoint ', checkpoint_path )
        model.eval();
    else :
        print(f'No checkpoint name provided, using pretrained {base_model} model weights')
    all_result = {}


    

    all_result['class_name'] =  eval_prompts(model=model,tokenizer=tokenizer,
                                prompt_dict= EUNIS_id_to_cls ,
                                visual_features_path=visual_features_path )

    all_result['aerial_dict'] =  eval_prompts(model=model,tokenizer=tokenizer,
                                prompt_dict= aerial_dict ,
                                visual_features_path=visual_features_path )

    all_result['rs_dict'] =  eval_prompts(model=model,tokenizer=tokenizer,
                            prompt_dict= rs_dict ,
                            visual_features_path=visual_features_path )

    all_result['sat_dict'] =  eval_prompts(model=model,tokenizer=tokenizer,
                                prompt_dict= sat_dict ,
                                visual_features_path=visual_features_path )
            
    all_result['short_name'] =  eval_prompts(model=model,tokenizer=tokenizer,
                                    prompt_dict= EUNIS_id_to_short_labels ,
                                    visual_features_path=visual_features_path )
    
    all_result['long_descr'] =  eval_prompts(model=model,tokenizer=tokenizer,
                                    prompt_dict= EUNIS_id_to_description ,
                                    visual_features_path=visual_features_path )
 

    outname = checkpoint_path.replace('.pt','_prompt_eval.csv')
    print(all_result)
    _=pd.DataFrame(all_result).to_csv(outname)
    print(f'file saved as {outname}')

def find_all_pt_files(output_dir):
    all_pt_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".pt") and 'features' not in file :
                folder = os.path.join(root, file)
                all_pt_files.append(folder)
                #load_model_and_dataset(output_dir, folder)
    return all_pt_files


if __name__ =="__main__":  



    root =  '/path/to/checkpoints/'

    
    all_pt_files = find_all_pt_files(root)

    
    print(f'found {len(all_pt_files)} .pt files')                  
    
    print('-'*150)
    for pt_file in reversed(all_pt_files) :
            
        folder = os.path.dirname(pt_file)
        checkpoint_name = pt_file.rsplit('/')[-1].replace('.pt','')
        
        print(folder,checkpoint_name)

        # Get model and args from the config file 
        if not os.path.exists(folder+'/args.txt') :
            continue
        args = read_config_file(folder+'/args.txt')

        generate_visual_features(folder, base_model=args['model'], checkpoint_name= checkpoint_name)
        run_model_evaluation (folder, base_model=args['model'], checkpoint_name= checkpoint_name)
    
        print('-'*150)
    print('Evaluation done for all models')
    