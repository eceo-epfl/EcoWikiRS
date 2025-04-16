import os 
ROOT = os.getcwd() + '/'



# images folder: 
SPLIT_PATH = ROOT + 'split'
IMAGE_DIR = '/data/swissimage/'  
# text folder :
WIKI_DIR = '/data/wiki/' 


NUM_SENTENCES_PER_SAMPLES  = 15

EUNIS_id_to_cls = {
    14: 'Surface standing waters',
    15: 'Surface running waters',
    16: 'Littoral zone of inland surface waterbodies',
    21: 'Mires, bogs and fens',
    23: 'Dry grasslands',
    24: 'Mesic grasslands',
    25: 'Seasonally wet and wet grasslands',
    26: 'Alpine and subalpine grasslands',
    31: 'Arctic, alpine and subalpine scrub',
    32: 'Temperate and mediterranean-montane scrub',
    40: 'Shrub plantations',
    41: 'Broadleaved deciduous woodland',
    43: 'Coniferous woodland',
    44: 'Mixed deciduous and coniferous woodland',
    45: 'Lines of trees, small anthropogenic woodlands, recently felled woodland, early-stage woodland and coppice',
    47: 'Screes',
    48: 'Inland cliffs, rock pavements and outcrops',
    49: 'Snow or ice-dominated habitats',
    50: 'Miscellaneous inland habitats with very sparse or no vegetation',
    52: 'Arable land and market gardens',
    53: 'Cultivated areas of gardens and parks',
    54: 'Buildings of cities, towns and villages',
    55: 'Low density buildings',
    56: 'Extractive industrial sites',
    57: 'Transport networks and other constructed hard-surfaced areas'
 }

EUNIS_id_to_cls_id = {
    14: 0,      15: 1,      16: 2,      18: 3,
    21: 3,      23: 4,      24:5,      25: 6,
    26: 7,      31: 8,      32: 9,     40: 10,
    41: 11,     43: 12,     44: 13,     45: 14,
    47: 15,     48: 16,     49: 17,     50: 18,
    52: 19,     53: 20,     54: 21,     55: 22,
    56: 23,     57: 24,
 }