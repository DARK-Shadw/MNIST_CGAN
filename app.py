import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
import os # Import os to check for file existence

# --- Actual Generator Class ---
# User provided Generator class definition
class Generator(nn.Module):
    """
    Conditional GAN Generator for MNIST-like digits.
    """
    def __init__(self, latent_dim, num_classes, img_size, device="cpu"): # Added device argument
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.img_size = img_size
        self.init_size = img_size // 4 # Example calculation, adjust if your model uses ConvTranspose
        # Define the model layers
        # Note: Ensure this architecture exactly matches the one used during training
        self.model = nn.Sequential(
            # Combine latent vector and label embedding
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True), # Changed ReLU to LeakyReLU based on common GAN practices, adjust if needed
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), # Removed momentum 0.8, use default or adjust if needed
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_size * img_size),
            nn.Tanh() # Output pixel values in [-1, 1]
        )
        # Note: Removed .to(device) here, will be applied after instantiation

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        # Ensure labels are long type for embedding
        gen_input = torch.cat((noise, self.label_emb(labels.long())), -1)
        # Generate flattened image data
        img_flat = self.model(gen_input)
        # Reshape to image dimensions (1 channel, img_size, img_size)
        img = img_flat.view(img_flat.size(0), 1, self.img_size, self.img_size)
        return img

# --- Image Generation Function (Modified for Streamlit) ---
def generate_continuous_digit_sequence(gen, sequence, latent_dim=100, device="cpu", save_path=None):
    """
    Generate a continuous image of a sequence of digits with a black background
    using a trained CGAN generator. Modified to return a matplotlib figure.

    Args:
        gen: Trained Generator model
        sequence: A list of digits or an integer/string
        latent_dim: Dimension of the latent space (default: 100)
        device: Device to run the model on (default: "cpu")
        save_path: If provided, saves the image to this path (default: None)

    Returns:
        matplotlib.figure.Figure: The figure containing the generated image, or None if error.
    """
    try:
        # Convert to list of digits if input is an integer or string
        if isinstance(sequence, int) or isinstance(sequence, str):
            sequence = [int(digit) for digit in str(sequence)]

        if not sequence:
            st.error("Input sequence is empty.")
            return None # Return None if sequence is empty

        # Ensure generator is on the correct device
        gen.to(device)
        gen.eval() # Set generator to evaluation mode

        # Generate one random noise vector per digit
        z = torch.randn(len(sequence), latent_dim).to(device)

        # Convert sequence to tensor
        labels = torch.tensor(sequence).to(device)

        # Generate images
        with torch.no_grad(): # Turn off gradients for inference
            images = gen(z, labels)
            images = images.cpu() # Move images to CPU for processing with numpy/matplotlib

        # Denormalize images from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Create a continuous image by concatenating the digits horizontally
        img_list = [img.squeeze().numpy() for img in images]
        if not img_list:
             st.error("Generated image list is empty.")
             return None

        # Determine image height from the first image for fallback
        img_height = img_list[0].shape[0] if len(img_list[0].shape) > 0 else 28 # Default height if squeeze removes all dims

        # Ensure concatenation axis=1, handle potential empty images
        continuous_img = np.concatenate(img_list, axis=1) if img_list else np.zeros((img_height, 0))

        # Create the plot
        # Adjust figsize dynamically based on sequence length and aspect ratio
        aspect_ratio = 1.0 # Assuming square digits
        fig_height = 1.5 # Fixed height in inches
        fig_width = len(sequence) * fig_height * aspect_ratio * 0.75 # Adjust multiplier as needed
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set figure and axes background to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Display the image
        ax.imshow(continuous_img, cmap='gray')
        ax.axis('off') # Hide axes

        plt.tight_layout(pad=0.1) # Adjust padding

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='black')
            print(f"Image saved to {save_path}")

        # Don't call plt.show() in Streamlit, return the figure instead
        return fig

    except Exception as e:
        st.error(f"Error during image generation: {e}")
        # Optionally log the full traceback for debugging
        # import traceback
        # st.error(traceback.format_exc())
        return None


# --- Streamlit App ---

# Set page configuration
st.set_page_config(page_title="GAN Digit Sequence Generator", layout="wide")

# --- App Layout ---
st.title("✍️ Handwritten Digit Sequence Generator")
st.markdown("Showcasing GAN learning progress. Enter a sequence of digits (like a phone number) to generate a corresponding handwritten image using a Conditional GAN.")

# --- Configuration ---
st.sidebar.header("Configuration")
# Model Parameters (Adjust if your model differs)
latent_dim = 100
num_classes = 10 # MNIST digits 0-9
img_size = 28    # MNIST image size
model_path = "generator_mnist.pt" # Path to your trained model file

# Device Selection
# Automatically detect CUDA availability
if torch.cuda.is_available():
    default_device = "cuda"
else:
    default_device = "cpu"
device = st.sidebar.selectbox("Select compute device", ["cuda", "cpu"], index=["cuda", "cpu"].index(default_device))

st.sidebar.info(f"Using device: `{device}`\nLatent Dimension: `{latent_dim}`\nImage Size: `{img_size}x{img_size}`")

# --- Load Your Model ---
@st.cache_resource # Cache the loaded model to avoid reloading on every interaction
def load_model(model_path, latent_dim, num_classes, img_size, device):
    """Loads the generator model."""
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found at: {model_path}")
        st.stop() # Stop execution if model file is missing

    try:
        # Instantiate your actual Generator class
        generator = Generator(latent_dim, num_classes, img_size, device=device) # Pass device here if needed by init
        # Load the state dictionary, mapping location to the selected device
        generator.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        # Move the model to the selected device (important!)
        generator.to(device)
        # Set to evaluation mode
        generator.eval()
        st.sidebar.success(f"Loaded model from `{model_path}`")
        return generator
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop() # Stop execution if model loading fails

# Load the generator
generator = load_model(model_path, latent_dim, num_classes, img_size, device)

# --- Input Section ---
st.header("Generate Image")
number_sequence = st.text_input(
    "Enter digit sequence (0-9):",
    placeholder="e.g., 1234567890",
    max_chars=30 # Increased max chars slightly
)

# --- Generate Button and Output ---
if st.button("Generate Image Sequence", type="primary"):
    if number_sequence:
        # Validate input: Check if it contains only digits
        if number_sequence.isdigit():
            with st.spinner(f'Generating image on {device}...'):
                # Call the generation function
                fig = generate_continuous_digit_sequence(
                    gen=generator,
                    sequence=number_sequence,
                    latent_dim=latent_dim,
                    device=device
                )

                if fig:
                    # Display the generated image
                    st.subheader("Generated Image:")
                    st.pyplot(fig)

                    # Optional: Add download button for the image
                    buf = io.BytesIO()
                    # Save figure to buffer
                    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1, facecolor='black')
                    buf.seek(0) # Rewind buffer
                    st.download_button(
                        label="Download Image",
                        data=buf, # Pass the buffer directly
                        file_name=f"gan_sequence_{number_sequence}.png",
                        mime="image/png"
                    )
                    # Close the figure to free memory
                    plt.close(fig)
                else:
                    # Error messages are now handled within the generation function
                    st.warning("Image generation failed. Check error messages above.")

        else:
            # Show error if input is invalid
            st.error("Invalid input. Please enter only digits (0-9).")
    else:
        # Show warning if input is empty
        st.warning("Please enter a digit sequence.")

st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io) and [PyTorch](https://pytorch.org).")
