import numpy as np

class DimReduceNet:
    """
    A deterministic, NumPy-based dimensionality reduction network using PCA (via SVD).
    Transforms high-dimensional input images into a lower-dimensional latent space.
    """
    def __init__(self, latent_dim=16):
        self.latent_dim = latent_dim
        self.mean_ = None
        self.components_ = None
        self.original_shape_ = None

    def fit(self, images):
        """
        Fits the model to the images by computing the mean and principal components.
        
        Args:
            images (np.ndarray): Array of shape (N, H, W) or similar.
            
        Returns:
            np.ndarray: The encoded latent representation of shape (N, latent_dim).
        """
        images_array = np.asarray(images)
        self.original_shape_ = images_array.shape[1:]
        
        # Flatten the images to (N, features)
        n_samples = images_array.shape[0]
        flattened = images_array.reshape(n_samples, -1)
        
        # Compute mean
        self.mean_ = np.mean(flattened, axis=0)
        
        # Center the data
        centered = flattened - self.mean_
        
        # Compute SVD. Use full_matrices=False for memory efficiency
        # We only need the top latent_dim singular vectors.
        # However, for svd, U is (N, N) or (N, K), S is (min(N, features),), Vh is (min(N, features), features)
        # Using SVD to compute PCA
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        
        # Store top latent_dim components
        # Vh contains the principal axes in rows
        self.components_ = Vh[:self.latent_dim]
        
        # Return encoded data
        return self.encode(images)

    def encode(self, images):
        """
        Projects images into the latent space.
        
        Args:
            images (np.ndarray): Array of shape (N, H, W).
            
        Returns:
            np.ndarray: Encoded latent representation of shape (N, latent_dim).
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model must be fitted before calling encode().")
            
        images_array = np.asarray(images)
        n_samples = images_array.shape[0]
        flattened = images_array.reshape(n_samples, -1)
        
        centered = flattened - self.mean_
        
        # Project data onto components: X_pca = (X - mean) @ V.T
        latent = np.dot(centered, self.components_.T)
        return latent

    def decode(self, latent):
        """
        Reconstructs images from the latent representation.
        
        Args:
            latent (np.ndarray): Array of shape (N, latent_dim).
            
        Returns:
            np.ndarray: Reconstructed images, reshaped to match original input shape.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model must be fitted before calling decode().")
            
        latent_array = np.asarray(latent)
        n_samples = latent_array.shape[0]
        
        # Reconstruct: X_rec = (latent @ V) + mean
        reconstructed_flat = np.dot(latent_array, self.components_) + self.mean_
        
        # Reshape back to original dimensions
        reconstructed = reconstructed_flat.reshape(n_samples, *self.original_shape_)
        return reconstructed

    def reconstruction_error(self, original, reconstructed):
        """
        Computes the Mean Squared Error (MSE) between original and reconstructed images.
        """
        orig = np.asarray(original)
        recon = np.asarray(reconstructed)
        return np.mean((orig - recon)**2)

    def save(self, filepath):
        """
        Saves the model parameters (mean, components, original shape) to a file.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model must be fitted before saving.")
            
        np.savez(filepath, 
                 latent_dim=self.latent_dim,
                 mean=self.mean_, 
                 components=self.components_,
                 original_shape=self.original_shape_)

    def load(self, filepath):
        """
        Loads the model parameters from a file.
        """
        data = np.load(filepath)
        self.latent_dim = int(data['latent_dim'])
        self.mean_ = data['mean']
        self.components_ = data['components']
        self.original_shape_ = tuple(data['original_shape'])
