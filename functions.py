import torch as pt
from os.path import isdir


train_loss = pt.zeros(10000)
val_loss = pt.zeros(10000)

#######################################################################################
class FirstNN(pt.nn.Module):
    """Simple fully-connected neural network.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_inputs = kwargs.get("n_inputs", 1)
        self.n_outputs = kwargs.get("n_outputs", 1)
        self.n_layers = kwargs.get("n_layers", 1)
        self.n_neurons = kwargs.get("n_neurons", 10)
        self.activation = kwargs.get("activation", pt.sigmoid)
        self.layers = pt.nn.ModuleList()
        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers-1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers)-1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)

    @property
    def model_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
#######################################################################################        


def optimize_model(model: pt.nn.Module, x_train: pt.Tensor, x_val: pt.Tensor, epochs: int=1000,
                   lr: float=0.001, save_best: str=""): 
 
    y_train = pt.zeros([200,10])
    print ("y_train_shape", y_train.shape)
    y_val = pt.zeros([10,10])
    
    print ("y_val.shape", x_val.shape)
    for i in range (len(x_train)-1):
    		y_train[i] = x_train[i+1]
    for i in range (len(x_val)-1):
    		y_val[i] = x_val[i+1]
    
    print("xtrain",x_train.shape)
    print("ytrain",y_train.shape)
    criterion = pt.nn.MSELoss()
    
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    train_loss, val_loss = [], []
    for e in range(1, epochs+1):
        optimizer.zero_grad()
        prediction = model(x_train).squeeze()  # x_train= ein Zeitschritt
        loss = criterion(prediction, y_train)	# y_train ist der nächste Zeitschritt
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        with pt.no_grad():
            prediction = model(x_val).squeeze()
            loss = criterion(prediction, y_val)
            val_loss.append(loss.item())
        print("\r", "Training/validation loss epoch {:5d}: {:10.5e}, {:10.5e}".format(
            e, train_loss[-1], val_loss[-1]), end="")
        if isdir(save_best):    
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_train.pt")
                best_train_loss = train_loss[-1]
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_val.pt")
                best_val_loss = val_loss[-1]
    return train_loss, val_loss


#######################################################################################

