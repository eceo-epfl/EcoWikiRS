import os,sys
import utils.variables as variables
from utils.variables import NUM_SENTENCES_PER_SAMPLES
sys.path.append(variables.ROOT)
os.chdir(variables.ROOT)

import csv
import json
import torch
import numpy as np
import random 
import torchvision.transforms as T
from torch.utils.data import Dataset 
from tifffile import tifffile

from random  import randint 



class WikiRSDataset(Dataset ) :
    """
    A PyTorch dataset class for handling the WikiRS dataset. 
    Returns 1 image, 1 or several senteces and eunis label.         
    """
    @torch.no_grad()
    def __init__(self, 
                 phase='train', 
                 debug=False,
                 article_key = 'habitat',
                 use_sentence_parts=False,  
                 one_sentence_per_item=False,
                 verbose =True,                        
                 ) -> None:
        """
        Initializes the dataset with the given parameters:
        Args:
            phase (str, optional): phase (str): The phase of the dataset, either 'train','test', 'val' or 'all' (for all data).
            debug (bool, optional): If True, limits the dataset is limited to the first 2000 items. Defaults to False.
            article_key (str): Key used to extract sentences from JSON files, from 'habitat','random' or 'keywords'.   
            use_sentence_parts (bool, optional): Uses sentence substring parts in the dataset. 
            one_sentence_per_item (bool, optional): Includes only one sentence per image.
            verbose (bool, optional): If True, prints detailed information about the dataset. Defaults to True.
        """        
        super().__init__()
        self.phase = phase
        self.verbose = False   
        self.use_sentence_parts = use_sentence_parts    
        self.one_sentence_per_item = one_sentence_per_item

        self.img_dir= variables.IMAGE_DIR
        self.wiki_dir = variables.WIKI_DIR
        assert article_key in ["habitat", "keywords" ,"random","species"], f"Invalid article_key: {article_key}. Choose from 'habitat', 'keywords', 'random', or 'species'."
        key_dict = {"habitat"   :"habitat_section", 
                    "keywords"  :"keywords" ,
                    "random"    :"random_sentences",
                    "species"   :"binomial_name",
                      }
        self.article_key = key_dict[article_key]     

        
        # Build dataset from file :
        self.id_list,self.species_dict,self.eunis_dict = self.read_dataset(split_path=variables.SPLIT_PATH)
        
        if debug : # Use a subset for debugging
            self.id_list = self.id_list[:2000]

        # Define COLOR TRANSFORM  
        self.augm= self.get_transform()

        # Print Dataset Description : 
        if verbose :           
            self.print_verbose(variables.SPLIT_PATH,debug)

    def __len__(self):
        return len(self.id_list)   
    
    def print_verbose(self,split_path,debug):
            print('-'*50) 
            print(  'Get dataset SPLIT from', split_path, self.phase)
            print( '\tWith',len(self.id_list), 'samples')
            print('\tUse species text from ', self.wiki_dir, 'with', self.article_key)   
            if self.use_sentence_parts :
                 print('\tSample 3-15 words from full sentence.')
            if debug : 
                print(f'\tDebug mode, only use {len(self.id_list)} samples')         
            if self.one_sentence_per_item:
                print('\tDataset with one random sentence per images', self.one_sentence_per_item)

    def read_dataset(self,split_path):
        """
        Reads the dataset split file based on self.phase and returns the ID list, species dictionary, and EUNIS dictionary.
        Args:
            split_path (str): path to sample list
        Returns:
            id_list (list): List of sample IDs.
            species_dict (dict): Dictionary mapping sample IDs to species information.
            eunis_dict (dict): Dictionary mapping sample IDs to EUNIS codes.
        """

        # Read list of tile id from file:      
        file = open(split_path+'/'+self.phase+'.csv', "r")
        data = list(csv.reader(file, delimiter=","))
        data = data[1:] #ignore first line with header
        file.close()  
       
        id_list = [x[0] for x in data]
        species_dict = { x[0]:x[3] for x in data}
        eunis_dict = { x[0]:x[1] for x in data}

        return id_list,species_dict,eunis_dict

    def get_transform(self):
        """ Defines and returns the image transformation pipeline based on the dataset phase."""
        
        if self.phase == 'train':  
            # Apply image augmentations in train phase only        
            augm = T.Compose([    
                    T.RandomResizedCrop   (size=224,scale=(0.2, 1.), interpolation=T.InterpolationMode.BICUBIC),                                 
                    RGBJitter(0.1),
                    T.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711], ),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),  
                    T.RandomRotation(5),                   
                    ])  
                        
        else : #validation or test
            augm =  T .Compose([
                    T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
                    ])
        return augm
    
    def get_img(self,id):
        """ Loads and preprocesses an image given its ID."""
        rgb_path = self.img_dir +  id +'.tif'
        rgb = tifffile.imread(rgb_path)
        rgb = np.moveaxis(rgb,-1,0)
        rgb = torch.from_numpy(rgb/255).float()

        # apply augmentation on input image
        rgb = self.augm(rgb)  
        return rgb
    
    def get_sentences(self,species_list):
        """
        Retrieves  sentences for a given species list, with 30%  random dropout.
        Return NUM_SENTENCES_PER_SAMPLES senteces per samples. Use padding if number of sentences is smaller.
        """
        sentences = []
        for sp in eval(species_list) :
            sp = sp.replace(' ','_').replace('\\','')
            with open(self.wiki_dir +  sp +'.json', 'r') as file:
                data = json.load(file)
                sentences += data[ self.article_key]

  
       # Randomly dropout 30% of the sentences :        
        random.shuffle(sentences)  # inplace operation
        end = len(sentences) //3 
        sentences = sentences[end:]
            
        # Get the correct number of sentences for a sample
        if len(sentences) > NUM_SENTENCES_PER_SAMPLES :
            sentences = random.sample(sentences,k=NUM_SENTENCES_PER_SAMPLES )

        elif len(sentences) < NUM_SENTENCES_PER_SAMPLES  :
           # Padding with empty string if fewer than NUM_SENT_PER_SAM is present
           sentences += ['']*( NUM_SENTENCES_PER_SAMPLES  -len(sentences))
            
        assert isinstance(sentences,list)
        assert len(sentences) == NUM_SENTENCES_PER_SAMPLES 
        sentences =  '<sep>'.join(sentences)
        return sentences

    def get_species_sentences(self,sp):
        """
        Retrieves  all sentences for a given species .
        """
        sentences = []
        sp = sp.replace(' ','_').replace('\\','')
        with open(self.wiki_dir +  sp +'.json', 'r') as file:
            data = json.load(file)
            sentences += data[ self.article_key]
        return sentences
    
    def get_sentences_substrings(self,species_list ):
        """Implementation of sentence substring augmentation. 
            Extracts and returns 3-15 words per sentences, for a given species list.
            """
        sentences = []
        for sp in eval(species_list) :
            sp = sp.replace(' ','_').replace('\\','')
            with open(self.wiki_dir +  sp +'.json', 'r') as file:
                data = json.load(file)
                sentences += data[ self.article_key]

        
        sentences_part = extract_word_groups(sentences)
        return '<sep>'.join(sentences_part )
    
    def get_one_sentence(self,species_list):
        """Retrieves one random sentence for a given species list."""

        sentences = []
        for sp in eval(species_list) :
            sp = sp.replace(' ','_').replace('\\','')
            with open(self.wiki_dir +  sp +'.json', 'r') as file:
                data = json.load(file)
                sentences += data[ self.article_key]

            
        # Randomly get 1 sentences
        if len(sentences) >=1 : #, f'sentences is too short {species_list} {sentences}'
            return  random.sample(sentences,k=1)
        else : 
            return []
        
    def __getitem__(self, index) :
        """Retrieves a single sample from the dataset, including image, text, and eunis label."""

        id = self.id_list[index]
        eunis = self.eunis_dict[id]
        data = {'id':id, 'eunis':int(eunis)}
                   
        # Get image :    
        data['image'] = self.get_img(id )          
        
        #Get species list :
        species_list = self.species_dict[id]
        data['species'] = species_list

        # Get sentences :
        if self.use_sentence_parts :
            sentences = self.get_sentences_substrings(species_list)
        elif self.one_sentence_per_item:
            sentences = self.get_one_sentence(species_list)
        else :
            sentences = self.get_sentences(species_list)
        data['text'] = sentences

        if not self.one_sentence_per_item:
            # return a padding boolean mask
            data['text_padding'] = torch.Tensor( [  k=='' for k in sentences.rsplit('<sep>')])
            
        return   data
    


class RGBJitter(object):
    def __init__(self, color_jitter_factor ):
        """ Apply color augmentation on RGB channels, i.e. channels 0,1,2.
            Input image values must be within [0,1].
        """
        
        self.color_jitter = T.ColorJitter(brightness= color_jitter_factor, 
                                          contrast= color_jitter_factor, 
                                          saturation = color_jitter_factor, 
                                          hue= 0.1,
                                          )

    def __call__(self, img):
        # Get the RGB channels
        rgb = img[:3, :, :]
     
        # Apply the color jitter to the RGB channels
        rgb = self.color_jitter(rgb)
        
        # Combine the jittered RGB channels with the remaining channels
        img_jittered = torch.cat([rgb, img[3:, :, :]], dim=0)
         
        return img_jittered
    

def extract_word_groups(sentences, group_count=NUM_SENTENCES_PER_SAMPLES, min_words=3, max_words=15):
    """Extract sub sentences from the sentence given as input
    Args:
        sentences (list of string): wikipedia sentences
        group_count (int): number of sentences to return.
        min_words (int, optional): _description_. Defaults to 3.
        max_words (int, optional): _description_. Defaults to 15.

    """
    word_groups = []

    while len(word_groups) < group_count:
        # Randomly choose a sentence
        if len(sentences ) >= 1:
            sentence = random.choice(sentences)
        else : 
            print('WEIRD len sent  less than 1', sentences) 
            return ''
            
        words = sentence.split()  # Split the sentence into words
        
        # Ensure that the sentence has enough words to extract a group
        if len(words) < min_words:
            word_groups.append(sentence)
            continue
        
        # Choose a random starting point, ensuring enough words remain for the max group size
        start_idx = random.randint(0, len(words) - min_words)
        
        # Randomly choose the length of the group between min_words and max_words
        group_length = random.randint(min_words, min(max_words, len(words) - start_idx))
        
        # Extract the word group
        word_group = words[start_idx:start_idx + group_length]
        
        # Join the words into a group and add to the list
        word_groups.append(' '.join(word_group))


    return word_groups

if __name__ =="__main__":  
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    dico = {
            'phase':'all',
            'debug' : True,            
            'article_key' : 'species',# 'habitat',
            'use_sentence_parts': False,
            'one_sentence_per_item':False, 

    }


    if True:    
        for phase in ['all']:
            for key in ['habitat','species','keywords']:
                dico['phase'] =phase
                dico['article_key'] =key
                ds=  WikiRSDataset( 
                                **dico
                                )
    

                dl = DataLoader(dataset=ds,
                                batch_size=32,
                                num_workers=20,
                                shuffle=True,
                                )
                
                for data in tqdm (dl) :
                    pass

                print('done',phase)


    print('terminated without error')