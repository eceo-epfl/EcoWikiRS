import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support,accuracy_score


from utils.dataset import WikiRSDataset
from utils.eunis import EUNIS_id_to_cls,EUNIS_id_to_cls_id



def eval_zero_shot_eunis(model, tokenizer, args=None,  device=None):
    """
        Evaluates a zero-shot model on the EUNIS habitat classification task.
        Args:
            model: The model to be evaluated, supporting text and image encoding.
            tokenizer: Tokenizer for processing text inputs.
            args (optional): arguments from argparser.
            device (optional): Device to run the evaluation on ('cuda' or 'cpu').
        Returns:
            dict: A dictionary containing evaluation metrics:
                - 'precision': Macro-averaged precision.
                - 'recall': Macro-averaged recall.
                - 'f1': Macro-averaged F1 score.
                - 'overall_acc': Overall accuracy of predictions.
    """
    if device ==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get dataloaders :       
    test_ds = WikiRSDataset(phase = 'val',verbose=False)
    test_loader   = DataLoader(dataset=test_ds,
                            batch_size=256,
                            num_workers=16,
                            shuffle=False,
                            drop_last=False,
                            )
    
    # Load EUNIS habitat definition from file
    eunis_description = list(EUNIS_id_to_cls.values())    

    model.eval();
    model.to(device);
    overall_accuracy, mean_metrics =[],[]

    with torch.no_grad(),torch.autocast(device_type=device):

        # Encode ecosystem description
        text_tokens =   tokenizer ( eunis_description) .to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features/ text_features.norm(dim=-1, keepdim=True)
        
        for  input in test_loader :
            # move data to device
            image_features = model.encode_image(input['image'].to(device))
            image_features = image_features/ image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T 
            logits = similarity.softmax(axis=0)
            prediction = torch.argmax(logits,axis=1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

            targets = input['eunis'].numpy()
            targets = [EUNIS_id_to_cls_id[int(x)] for x in targets]

            overall_accuracy += [accuracy_score(y_true=targets, y_pred = prediction)]
            mean_metrics += [precision_recall_fscore_support(y_true = targets, 
                                        y_pred = prediction, 
                                        average = 'macro',  
                                        zero_division = 0,                                                        
                                        )]

    overall_accuracy = np.stack(overall_accuracy).mean(axis=0)
    mean_metrics= np.stack(mean_metrics)[:,:-1].mean(axis=0)
    mean_metrics = np.array(mean_metrics,dtype ='float').round(3)


    
    return {'precision':mean_metrics[0],
            'recall':mean_metrics[1],
            'f1':mean_metrics[2], 
            'overall_acc':overall_accuracy.round(3),
            }


