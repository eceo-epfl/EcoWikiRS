
import os
import pandas as pd
import torch
from utils.utils import get_model

from utils.prompts_template import  EUNIS_id_2_ollama_descr, aerial_dict,rs_dict,sat_dict
from utils.eunis import  EUNIS_id_to_cls, EUNIS_id_to_short_labels, EUNIS_id_to_description
from utils.utils import read_config_file
from utils.utils import read_config_file,get_model
from run_model_evaluation import generate_visual_features,  eval_prompts
from run_model_evaluation import find_all_pt_files

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_prompt_search(folder, base_model, checkpoint_name=None):
    """
    Executes a prompt search process for a given model and dataset, evaluating 
    various prompt dictionaries and saving the results to a CSV file.
    Args:
        folder (str): The directory containing the model checkpoints and visual features.
        base_model (str): The name of the base model to be used for evaluation among CLIP, SkyCLIP,GeoRSCLIP and RemoteCLIP.
        checkpoint_name (str, optional): The name of the checkpoint file (without extension) 
            to load fine-tuned model weights. If None, the base model's pretrained weights 
            are used. Defaults to None.
    Returns:
        None: The function saves the evaluation results to a CSV file and does not return anything.
    """
    
    # Look for the visual features:
    ################################  
    checkpoint_path = f"{folder}/{checkpoint_name}.pt"
    outname = checkpoint_path.replace('.pt','_prompt_search.csv')
    if os.path.exists (outname) :
        print('\t-->Evaluation metrics already computed : ',outname)
        return
    out_folder = folder + f'/visual_features/{checkpoint_name}'
    visual_features_path= out_folder+ "/train_features.pt"
    
    if not os.path.exists (visual_features_path) :
        generate_visual_features(folder, base_model, checkpoint_name,phase='train')
    
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

    best_25_ollama_prompt =  EUNIS_id_2_ollama_descr

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

    outname = checkpoint_path.replace('.pt','_prompt_search.csv')
    print(all_result)
    _=pd.DataFrame(all_result).to_csv(outname)
    print(f'file saved as {outname}')



if __name__ =="__main__":  

    root =  '/my/project/folder/'

    all_pt_files = find_all_pt_files(root)
            
    
    print('-'*150)
    print(f'found {len(all_pt_files)} .pt files')   
    for pt_file in all_pt_files :
            
        folder = os.path.dirname(pt_file)
        checkpoint_name = pt_file.rsplit('/')[-1].replace('.pt','')
        

        # Get model and args from the config file 
        args = read_config_file(folder+'/args.txt')
        

        print(folder,checkpoint_name)
        
        run_prompt_search(folder, base_model=args['model'], checkpoint_name= checkpoint_name)

    
        print('-'*150)
    print('Evaluation done for all models')
    