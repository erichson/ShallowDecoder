import torch
from torch import nn
from torch.autograd import Variable, Function


def model_from_name(name, outputlayer_size, n_sensors):
    if name == 'shallow_decoder':
        return shallow_decoder(outputlayer_size, n_sensors)    

    elif name == 'shallow_decoder_drop':
        return shallow_decoder_drop(outputlayer_size, n_sensors)    

    raise ValueError('model {} not recognized'.format(name))
    

class shallow_decoder(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(shallow_decoder, self).__init__()
        
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size
        
        self.learn_features = nn.Sequential(         
            nn.Linear(n_sensors, 40),
            nn.ReLU(True), 
            nn.BatchNorm1d(1),  
            )        
        
        self.learn_coef = nn.Sequential(            
            nn.Linear(40, 45),
            nn.ReLU(True),  
            nn.BatchNorm1d(1),  
            )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(45, self.outputlayer_size),
            )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)    


    def forward(self, x):
        x = self.learn_features(x)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x) 
        return x


    
class shallow_decoder_drop(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(shallow_decoder_drop, self).__init__()
        
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size
        
        
        self.learn_features = nn.Sequential(         
            nn.Linear(n_sensors, 40),
            nn.ReLU(True),   
            nn.BatchNorm1d(1),  
            )        
        
        self.learn_coef = nn.Sequential(            
            nn.Linear(40, 45),
            nn.ReLU(True),  
            nn.BatchNorm1d(1),  
            )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(45, self.outputlayer_size),            
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)     
                    
        
    def forward(self, x):
        x = self.learn_features(x)
        x = nn.functional.dropout(x, p=0.1, training=self.training)       
        x = self.learn_coef(x)
        x = self.learn_dictionary(x) 
        return x   
