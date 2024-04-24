import torch
import torchvision.models as models

# PyTorch models store the learned parameters in an internal state dictionary,
# called state_dict. These can be persisted via the torch.save method:
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to
# Users/kevin/.cache/torch/hub/checkpoints/vgg16-397923af.pth


# To load model weights, you need to create an instance of the same model
# first, and then load the parameters using load_state_dict() method.
#
# NOTE: be sure to call model.eval() method before inferencing to set the
# dropout and batch normalization layers to evaluation mode. Failing to do
# this will yield inconsistent inference results.
model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# When loading model weights, we needed to instantiate the model class first,
# because the class defines the structure of a network. We might want to save
# the structure of this class together with the model, in which case we can
# pass model (and not model.state_dict()) to the saving function:
torch.save(model, '../models/model.pth')

# We can then load the model like this:

model = torch.load('../models/model.pth')
