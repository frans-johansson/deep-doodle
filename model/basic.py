"""Basic SketchRNN model implementation"""

import torch
import torch.nn as nn
from model.components import Decoder, Encoder
from model import device
from utils.data import DataAugmentation
from utils.sampling import sample_stroke


class SketchRNN(nn.Module):
    """
    An implementation of the basic VAE model presented in the source paper
    'A Neural Representation of Sketch Drawings'
    """

    def __init__(self, enc_hidden, dec_hidden, z_dims, num_mixtures, dropout):
        """
        Initializes a SketchRNN model with given parameters using the
        `Encoder`, `Sampler` and `Decoder` components from the model.components
        module

        Args:
            enc_hidden: Dimensions of the hidden latent space H which the input sketches are mapped to
                in the decoder
            dec_hidden: Dimensions of the hidden latent space used by the decoder LSTM which is then mapped
                to the output mixture model parameters
            z_dims: Dimensions of the hidden latent space Z which is fed to the decoder component
            num_mixtures: The number of mixture components to generate for each stroke
            dropout: Dropout keep probability
        """
        super(SketchRNN, self).__init__()

        self.augment = DataAugmentation()
        self.encoder = Encoder(
            hidden_size=enc_hidden,
            z_dims=z_dims,
            num_layers=2,
            dropout=dropout
        )
        self.decoder = Decoder(
            hidden_size=dec_hidden,
            z_dims=z_dims,
            num_layers=2,
            dropout=dropout,
            num_mixtures=num_mixtures,
        )

    def forward(self, S):
        """
        Encodes, samples and decodes one batch of strokes

        Args:
            S: Padded input batch of strokes with shape (batch, 5, seq_len)

        Returns:
            params, mu, sigma_hat: The GMM and pen parameters for each of the strokes along with the mean
                and unnormalized standard deviation of the latent space z.
        """
        S = self.augment(S)
        z, mu, sigma_hat = self.encoder(S)
        # Append the start of sequence vector to each item in the batch
        sos = torch.stack([torch.tensor([0, 0, 1, 0, 0])] * S.shape[0]).to(device)
        decoder_S = torch.cat([sos.unsqueeze(1), S], dim=1)
        params = self.decoder(z, decoder_S)

        return params, mu, sigma_hat

    def conditional_sample(self, input):
        """
        Generates one conditional sample from the model given a single input sketch.

        Args:
            input: A single input sketch, must be of shape (1, seq_len, 5) i.e. with a batch size of 1.

        Returns:
            A tensor of shape (1, seq_len, 5) with the sampled results in stroke-5 format.
        """
        self.encoder.eval()
        self.decoder.eval()
        
        N = input.shape[1]  # Sequence length
        z, _, _ = self.encoder(input)  # Encode the input
        
        stroke = torch.tensor([0, 0, 1, 0, 0])
        h_c = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        samples = [stroke]

        with torch.no_grad():
            for _ in range(N):
                S = stroke.unsqueeze(0).unsqueeze(0)
                params, h_c = self.decoder(z, S, h_c)
                params = tuple([param.squeeze(0).squeeze(0) for param in params])
                stroke = sample_stroke(*params).to(torch.float32)
                samples.append(stroke)

        return torch.stack(samples)
