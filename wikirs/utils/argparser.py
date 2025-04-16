from argparse import ArgumentParser
import argparse
from datetime import datetime

def generate_experiment_name(args, parser):
    """
    Generate a unique name for the experiment based on non-default argument values.
    """
    name_parts = [datetime.now().strftime("%Y%m%d_%H-%M")]

    # Get default values for all arguments
    defaults = {key: parser.get_default(key) for key in vars(args)}

    # Append only arguments that differ from their default value
    for key, value in vars(args).items():
        if value != defaults[key]:
            name_parts.append(f"{key}_{value}")

    # Join parts into a single name
    if len(name_parts) ==1:
        name_parts.append('base')
    outname = "_".join(name_parts)
    outname = outname.replace('_model','').replace('_text','').replace('temperature','temp').replace('weight_decay','wd')
    outname = outname.replace('_debug_True','').replace('bootstrap_beta','').replace('_criterion','').replace('Weighted_InfoNCELoss','WINCEL')

    return outname


def parse_args ():
   
    parser = ArgumentParser(description='Train WikiRS')

    ## General 
    ######################
    parser.add_argument('--debug', 
                        action=argparse.BooleanOptionalAction,
                        help='Enable debugging mode',
                        default=False,
                        )   
    parser.add_argument('--model',
                        help='choose pretrained model among CLIP, SkyCLIP, RemoteCLIP and GeoRSCLIP',
                        default='SkyCLIP',
                        type=str)
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--epoch',
                        help='number of training epochs',
                        default=60,
                        type=int)
    parser.add_argument('--bs',
                        help='batch size',
                        default=256,
                        type=int)
    parser.add_argument('--temperature',
                        help='temperature for the contrastive loss',
                        default=0.15,
                        type=float) 
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=1e-4,           
                        type=float) 
    parser.add_argument('--bootstrap_beta',
                        help='beta value for both soft and hard bootstraping, beta in ]0,1], with beta = 0.9 means mostly rely on targets/logits',
                        default=1.,           
                        type=float) 
    parser.add_argument('--criterion',
                        help='loss function, among wincel, infonce, hard_bootstrap, soft_bootstrap, sample_top_p, top_1',
                        default='wincel',
                        type=str)
    parser.add_argument('--text',
                        help='article section from "habitat", "random", "species" names or "keywords" sentences',
                        default="habitat",
                        type=str)  
    parser.add_argument('--use_sentence_parts', 
                        action=argparse.BooleanOptionalAction,
                        help='Enable dataset with sentences subparts',
                        default=False,
                        )  
    parser.add_argument('--log_wandb', 
                        action=argparse.BooleanOptionalAction,
                        help='Enable dataset with sentences subparts',
                        default=True,
                        )  
                        
    args = parser.parse_args()
    
    # Generate the experiment name
    args.name = generate_experiment_name(args, parser)
    args.num_workers = 16
    
    # Modify some arguments for debug mode
    if args.debug :
        args.name = 'debug'
        args.log_wandb = False
        args.epoch = 3

    return args 