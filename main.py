from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import os
import base64
from vlm import *
import traceback
import tempfile

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model and tokenizer globally
path = 'OpenGVLab/InternVL2_5-1B'
device_map = split_model('InternVL2_5-1B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        # Check if image and prompt are present
        if 'image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing image or prompt in request'}), 400
        
        # Extract base64 image and prompt
        base64_image = data['image']
        prompt = data['prompt']
        
        print(prompt , base64_image)
        print(f"Received prompt: {prompt}")
        
        # Remove the data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image.save(temp_file.name, format='JPEG')
                temp_file_path = temp_file.name
            
            # Process image using VLM model
            try:
                # Convert image to tensor using the temporary file
                pixel_values = load_image(temp_file_path, max_num=12).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=1024, do_sample=True)
                # Add image token to prompt
                base_prompt = """
                    You are a assistant tasked with providing emotional support to the person who is in need of it.
                    This is a picture of myself.
                """
                full_prompt = f'{base_prompt}\n<image>\n{prompt}'
                
                # Get response from model
                response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
                print(f"Model response: {response}")
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
                return jsonify({
                    'status': 'success',
                    'prompt': prompt,
                    'response': response
                })
                
            except Exception as e:
                # Clean up the temporary file in case of error
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
                error_traceback = traceback.format_exc()
                print(f"Error processing image: {str(e)}\n{error_traceback}")
                return jsonify({
                    'error': f'Error processing image: {str(e)}',
                    'traceback': error_traceback
                }), 500
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error decoding image: {str(e)}\n{error_traceback}")
            return jsonify({
                'error': f'Error decoding image: {str(e)}',
                'traceback': error_traceback
            }), 400
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Server error: {str(e)}\n{error_traceback}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'traceback': error_traceback
        }), 500

if __name__ == '__main__':
    app.run(debug=False)
