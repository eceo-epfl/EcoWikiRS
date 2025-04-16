import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as dist


class CELoss(nn.Module):
    """
    Custom Cross Entropy Loss for classification tasks.

    Methods:
        forward(outputs, targets): Computes  the cross-entropy loss.
    """
    
    def __init__(self):
        super().__init__()             

    def forward(self, outputs, targets):        
        loss = F.cross_entropy(outputs,targets)
        return  loss
    


class Weighted_InfoNCE_Loss(nn.Module):
    """ 
    The Weighted InfoNCE Loss (WINCEL) function is designed for contrastive learning tasks, where it computes
    a weighted InfoNCE loss between image and text features. 
    It allows for weighted representations of text features based on their similarity to image features.
    Attributes:
        temperature (float): A scaling factor for the logits to control the sharpness
            of the softmax distribution.
        device (str): The device to run the computations on. Defaults to 'cuda:0' if
            a GPU is available, otherwise 'cpu'.
    
    forward(img_feat, text_feat):
        Computes WINCEL between image and text features.
        Args:
            img_feat (torch.Tensor): A tensor of shape (B, hdim) representing
                image features, where B is the batch size and hdim is the visual feature
                dimension.
            text_feat (torch.Tensor): A tensor of shape (B, n_sent, hdim) representing
                text features, where B is the batch size, n_sent is the number of sentences per sample, 
                and hdim the text features dimensions.
        Returns:
            torch.Tensor: The computed loss value as a scalar tensor.
    Example:
        >>> loss_fn = Weighted_InfoNCE_Loss(temperature=0.1)
        >>> img_feat = torch.randn(32, 512)  # Example image features
        >>> text_feat = torch.randn(32,15, 512)  # Example text features
        >>> loss = loss_fn(img_feat, text_feat)
        >>> print(loss)
    """ 
    def __init__(self,temperature,device=None):
        super().__init__()
        self.temperature = temperature
        print(f'Load WINCEL  with temperature {temperature} ')


        if device == None :
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
        else : 
            self.device  = device   


    def forward(self, img_feat,text_feat):

        B,_ = img_feat.shape
        assert text_feat.shape[0] ==B, f'text and image features should have the same batch size but have shape: {img_feat.shape} {text_feat.shape}'

        # Calculating the alpha_n_i weights (img to each sentence similarity ) : 
        # alpha shape : B X n_sent x 1
        VL_sim = torch.einsum('bij,bjk->bik',text_feat, img_feat.unsqueeze(-1))
        alpha = torch.softmax(VL_sim/self.temperature,dim=1)

        # Weighted representation of text features : 
        A_n = torch.sum ( alpha * text_feat, axis =1 )
        A_n  = A_n/ A_n.norm(dim=-1, keepdim=True)        # normalised text representation An 
        

        # Calculating the Loss
        logits = (A_n @ img_feat.T) / self.temperature
        targets = torch.eye(B).to(device=self.device)

        texts_loss = F.cross_entropy(logits, targets , reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 

        return loss.mean()



class sample_top_p_text_Loss(nn.Module):
    """
    A  loss function for sampling text features based on their similarity to image features. 
    The sampling probability is determined by the cross-modal similarity. 
    Methods:
        forward(img_feat, text_feat):
            Computes the loss given image features and text features. 
    Args:
        temperature (float): The temperature parameter for scaling the softmax distribution.
        device (str, optional): Defaults to 'cuda:0' if a GPU is available, otherwise 'cpu'.

    """ 
    def __init__(self,temperature,device=None):
        super().__init__()
        self.temperature = temperature
        print(f'Load sample top p text  with temperature {temperature}')


        if device == None :
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
        else : 
            self.device  = device   


    def forward(self, img_feat,text_feat):

        B,_ = img_feat.shape
        assert text_feat.shape[0] ==B, f'text and image features should have the same batch size but have {B} and {text_feat.shape[0]}'

        with torch.no_grad():
            VL_sim   = torch.einsum('bij,bjk->bik',text_feat, img_feat.unsqueeze(-1))
            probabilities  = torch.softmax(VL_sim/self.temperature,dim=1).squeeze()
            
            # Create a distribution object
            distribution = dist.Categorical(probabilities)

            # Sample from the distribution
            # Samples are integers from {0,…,K−1}{0,…,K−1} where K is probs.size(-1).
            indices = distribution.sample()   
            # Get the sampled text features A_n :   
            A_n = text_feat[torch.arange(B),indices] 
            A_n  = A_n/ A_n.norm(dim=-1, keepdim=True)      
        

        # Calculating the Loss
        logits = (A_n @ img_feat.T) / self.temperature
        targets = torch.eye(B).to(device=self.device)

        texts_loss = F.cross_entropy(logits, targets , reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 

        return loss.mean()


class sample_top_1_text_Loss(nn.Module):
    """
    A  loss function for sampling the text features with the highest similarity to the image features.  
    Methods:
        forward(img_feat, text_feat):
            Computes the loss given image features and text features. 
    Args:
        temperature (float): The temperature parameter for scaling the softmax distribution.
        device (str, optional): Defaults to 'cuda:0' if a GPU is available, otherwise 'cpu'.
    """
    def __init__(self,temperature,device=None):
        super().__init__()
        self.temperature = temperature
        print(f'Load top-1 text sampling loss with temperature {temperature}')

        if device == None :
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
        else : 
            self.device  = device   


    def forward(self, img_feat,text_feat):

        B,_ = img_feat.shape
        assert text_feat.shape[0] ==B, f'text and image features should have the same batch size but have {img_feat.shape} {text_feat.shape}'


        with torch.no_grad():
            VL_sim   = torch.einsum('bij,bjk->bik',text_feat, img_feat.unsqueeze(-1))

            # Get the  text features A_n with highest cross modal similarity:        
            max_index = torch.argmax(VL_sim ,dim=1).squeeze() 
            A_n = text_feat[torch.arange(B),max_index]    
            A_n  = A_n/ A_n.norm(dim=-1, keepdim=True)        
        

        # Calculating the Loss
        logits = (A_n @ img_feat.T) / self.temperature
        targets = torch.eye(B).to(device=self.device)

        texts_loss = F.cross_entropy(logits, targets , reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 

        return loss.mean()


class SoftBootstrap(nn.Module):
    """
    This implements the Soft Bootstraping (Reed et al., 2015) on top of the WINCEL loss.
    Soft bootstraping handles noisy labels by modifying the targets and adding a fraction of the predicted logits. 
    Attributes:
        temperature (float): A scaling factor for the logits to control the sharpness of the 
            softmax distribution.
        device (str): Defaults to 'cuda:0'  if a GPU is available, otherwise 'cpu'.
        bootstrap_beta (float): A weighting factor for combining the true labels with the model's 
            predictions. Must be in the range (0, 1). High values give more weight to the true labels.

    """
    def __init__(self,temperature,device=None, bootstrap_beta=1.):
        super().__init__()
        self.temperature = temperature
        print(f'Load Soft Bootstraping with temperature {temperature} and bootstrap beta {bootstrap_beta} ')

        self.bootstrap_beta = bootstrap_beta
        assert bootstrap_beta <1 and bootstrap_beta >0 , f'Bootstrap beta values should be in ]0,1[, but got {bootstrap_beta}.'

        if device == None :
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
        else : 
            self.device  = device   


    def forward(self, img_feat,text_feat):

        B,_ = img_feat.shape
        assert text_feat.shape[0] ==B, f'text and image features should have the same batch size but have {img_feat.shape} {text_feat.shape}'

        # Calculating the alpha_n_i weights (img to each sentence similarity ) : 
        # alpha shape : B X n_sent x 1
        VL_sim = torch.einsum('bij,bjk->bik',text_feat, img_feat.unsqueeze(-1))
        alpha = torch.softmax(VL_sim/self.temperature,dim=1)

        # Weighted representation of text features : 
        A_n = torch.sum ( alpha * text_feat, axis =1 )
        A_n  = A_n/ A_n.norm(dim=-1, keepdim=True)        # normalised text representation An 
        

        # Calculating the Loss
        logits = (A_n @ img_feat.T) / self.temperature
        # Soft bootstraping :
        labels  = torch.eye(B).to(device=self.device)
        soft_tgt =  logits.clone().detach() # best img for each text        
        targets = labels * self.bootstrap_beta + (1-self.bootstrap_beta) * soft_tgt       


        texts_loss = F.cross_entropy(logits, targets , reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')

        loss =  (images_loss + texts_loss) / 2.0 

        return loss.mean()

class HardBootstrap(nn.Module):
    """
    This implements the Hard Bootstraping (Reed et al., 2015) on top of the WINCEL loss.
    Hard bootstraping handles noisy labels by modifying the targets and adding a fraction of the predicted logits. 
    Attributes:
        temperature (float): A scaling factor for the logits to control the sharpness of the 
            softmax distribution.
        device (str): Defaults to 'cuda:0'  if a GPU is available, otherwise 'cpu'.
        bootstrap_beta (float): A weighting factor for combining the true labels with the model's 
            predicted target. Must be in the range (0, 1). High values give more weight to the true labels.
    """
    def __init__(self,temperature,device=None, bootstrap_beta=1.):
        super().__init__()
        self.temperature = temperature
        print(f'Load Hard Bootstraping with temperature {temperature} and bootstrap beta {bootstrap_beta} ')

        self.bootstrap_beta = bootstrap_beta
        assert bootstrap_beta <1 and bootstrap_beta >0 , f'Bootstrap beta values should be in ]0,1[, but got {bootstrap_beta}.'

        if device == None :
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
        else : 
            self.device  = device   


    def forward(self, img_feat,text_feat):

        B,_ = img_feat.shape
        assert text_feat.shape[0] ==B, f'text and image features should have the same batch size but have {img_feat.shape} {text_feat.shape}'

        # Calculating the alpha_n_i weights : 
        # alpha shape : B X n_sent x 1
        VL_sim = torch.einsum('bij,bjk->bik',text_feat, img_feat.unsqueeze(-1))
        alpha = torch.softmax(VL_sim/self.temperature,dim=1)

        # Weighted representation of text features : 
        A_n = torch.sum ( alpha * text_feat, axis =1 )
        A_n  = A_n/ A_n.norm(dim=-1, keepdim=True)        # normalised text representation An
        

        # Calculating the Loss
        logits = (A_n @ img_feat.T) / self.temperature
        # Hard bootstraping :
        hard_tgt =  torch.nn.functional.one_hot(torch.argmax(logits,dim=1),num_classes=B ) .detach() # best img for each text
        labels  = torch.eye(B).to(device=self.device)
        targets = labels * self.bootstrap_beta + (1-self.bootstrap_beta) * hard_tgt        

        texts_loss = F.cross_entropy(logits, targets , reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')

        loss =  (images_loss + texts_loss) / 2.0 

        return loss.mean()

class infoNCELoss(nn.Module):
    """
    Implementation of the InfoNCE loss, designed for single text and image features.
    Args:
        temperature (float, optional): A scaling factor for the logits. Default is 0.1.
        
    Attributes:
        temperature (float): The temperature scaling factor.
        device (str): The device on which the computation will run ('cuda:0' if GPU is available, otherwise 'cpu').

    Methods:
        forward(img_feat, text_feat):
            Computes the InfoNCE loss for each image and text pairs.
    Forward Args:
        img_feat (torch.Tensor): A tensor containing image features of shape (batch_size, feature_dim).
        text_feat (torch.Tensor): A tensor containing text features of shape (batch_size, feature_dim).
    Forward Returns:
        torch.Tensor: The computed InfoNCE loss as a scalar value.
    Example:
        >>> loss_fn = infoNCELoss(temperature=0.1)
        >>> img_feat = torch.randn(32, 128)  # Example image features
        >>> text_feat = torch.randn(32, 128)  # Example text features
        >>> loss = loss_fn(img_feat, text_feat)
        >>> print(loss)
    """
    
    def __init__(self,temperature=0.1 ):
        super().__init__()
        self.temperature = temperature
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

        print(f'Load infoNCE loss for single text  with temperature { temperature}')   

    def forward(self, img_feat,text_feat):

        logits = (text_feat @ img_feat.T) / self.temperature
        targets = torch.eye(logits.shape[0]).to(self.device)

        texts_loss = F.cross_entropy(logits, targets,  reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T,  reduction='none')

        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        return loss.mean()
