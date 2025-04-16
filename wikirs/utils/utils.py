import open_clip
import torch
import argparse
import utils.variables as variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_args_to_file(args, filename='args.txt'):
    """Save argparser variables to a text file."""
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

def unnormalize(image_tensor):
    """
    Unnormalize a tensor between 0-1
    """                 
    mean=[0.48145466, 0.4578275, 0.40821073] 
    std=[0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    unnormalized_image = image_tensor * std + mean
    unnormalized_image = torch.clamp(input=unnormalized_image,min=0,max=1)   
    
    return unnormalized_image


def read_config_file(file_path):
    # Read the text file and parse it into a dictionary
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Convert value to the appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)

                config[key] = value

    # Convert the dictionary to an argparse.Namespace
    return argparse.Namespace(**config).__dict__

def get_model (model_name, verbose = False):
    """Return model and tokenizer for CLIP, RemoteCLIP, GeoRSCLIP or SkyCLIP.
    Args:
        model_name (str): model name to load.
        verbose (bool: Print details about model loading. Defaults to False.

    Returns:
        model (nn.Module): The loaded model.
        tokenizer (Tokenizer): The tokenizer associated with the model.
    """
    assert model_name in ['CLIP','RemoteCLIP','GeoRSCLIP','SkyCLIP'], f'Model name {model_name} is not implemented'
    model, tokenizer  = None,None

    if model_name == 'GeoRSCLIP':
        print(variables .ROOT)
        ckpt_path =  variables .ROOT + "lib/GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt"
        model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        message = model.load_state_dict(checkpoint, strict=False)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    elif model_name == 'RemoteCLIP':
        ckpt = torch.load(variables .ROOT +"lib/RemoteCLIP/checkpoints/RemoteCLIP-ViT-B-32.pt", 
                          map_location=device,
                          weights_only=True
                          )        
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        message = model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        

    elif model_name == 'CLIP':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.to(device)
        message = 'CLIP pretrained'


    elif model_name == 'SkyCLIP':
        ckpt = torch.load(variables .ROOT + "lib/Skyscript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt", 
                            map_location=device,
                          weights_only=False
                            ) 
        state_dict = {}
        for k,v in ckpt['state_dict'].items():
            # Rename the layer to match open_clip  models :
            state_dict [k.replace('module.','')]=v
        
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', 
                                                            force_quick_gelu=True, 
                                                           # precision = 'amp'
                                                            )
        model.to(device)
        message = model.load_state_dict(state_dict)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    
    assert model is not None , 'Model was not initiated !! '  
    if verbose : 
        print('-'*50)
        print('Get Model from pretrained ',  model_name)  
        print(message)
        print('-'*50)
        print('Layer that requires grad :')

    ## Check which layer are training :        
    for name, param in model.named_parameters():          
        param.requires_grad  = False
        if any ([ k in name for k in  ['visual.ln_post', 'visual.positional_embedding']]) :
            param.requires_grad  = True
        if  param.requires_grad and verbose:
            print(name,param.requires_grad)
    model.to(device)
    return model, tokenizer 


