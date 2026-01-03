# TongueGPT: Multimodal LLM for Traditional Chinese Tongue Diagnosis


## ⚡ Introduction
Welcome to **TongueGPT**, a specialized multimodal large language model for Traditional Chinese Medicine (TCM) tongue diagnosis!


**TongueGPT** is a cutting-edge AI system designed specifically for TCM tongue diagnosis. Through extensive training on tongue images and TCM diagnostic knowledge, our model can:

- 🔍 **Intelligent Tongue Analysis**: Automatically identify tongue body, coating, shape, and other features
- 🩺 **TCM Pattern Differentiation**: Perform eight-principle pattern differentiation and organ system analysis based on tongue characteristics
- 💡 **Diagnostic Recommendations**: Provide personalized TCM health advice and treatment suggestions
- 📊 **Disease Progression Tracking**: Support comparative analysis of tongue images across multiple time points

## 📚 Specialized Tongue Diagnosis Datasets

We open-source large-scale datasets specifically constructed for tongue diagnosis tasks:

| Dataset | Size | Description | 
|---------|------|-------------|
| **Tongue Pretraining Dataset** | ~500K tongue images + diagnostic text | Includes various tongue types, lighting conditions, angles, covering common tongue features |
| **Tongue Instruction Dataset** | 20K dialogue pairs | Contains Q&A scenarios for tongue diagnosis, pattern analysis, health advice |
| **Tongue Vision Benchmark** | 5K finely annotated tongue images | For model evaluation, includes 36 expert-annotated features |

## 🏥 Models

### Model Access

| Model | Parameters | Modalities |
|-------|------------|------------|
| **TongueGPT-7B-VL** | 7B | Text, Image |
| **TongueGPT-13B-VL** | 13B | Text, Image |
| **TongueGPT-7B-LLM** | 7B | Text-only |

### Model Inference

#### A. Launch with Gradio Demo

```bash
pip install gradio torch transformers pillow
python demo/tongue_analysis_demo.py --model_path TongueGPT-7B-VL
```

#### B. Text-Only Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TongueGPT-7B-LLM",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TongueGPT-7B-LLM")

# TCM tongue diagnosis questions
questions = [
    "What pattern might pale white tongue body with thin white coating indicate?",
    "What does scalloped tongue edges and swollen tongue body typically suggest?",
    "What is the TCM pattern differentiation for red tongue with yellow greasy coating?"
]

for question in questions:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
```

#### C. Image-Text Inference (Tongue Analysis)

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model_path = "TongueGPT-7B-VL"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def analyze_tongue_image(image_path):
    """Analyze tongue image"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Construct conversation
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please analyze this tongue image in detail, including tongue body, coating, shape characteristics, and provide TCM pattern differentiation."}
        ]
    }]
    
    # Prepare input
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )
    
    # Decode output
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# Example usage
image_path = "path/to/tongue_image.jpg"
analysis_result = analyze_tongue_image(image_path)
print("Tongue Analysis Result:")
print(analysis_result)
```

#### D. Batch Tongue Analysis (For Clinical Research)

```python
import pandas as pd
from pathlib import Path

def batch_tongue_analysis(image_dir, output_csv="tongue_analysis_results.csv"):
    """Batch analyze tongue images"""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"Analyzing image {i}/{len(image_files)}: {img_path.name}")
        
        try:
            analysis = analyze_tongue_image(str(img_path))
            
            # Parse key information (customize as needed)
            result = {
                "image_name": img_path.name,
                "analysis": analysis,
                "tongue_color": extract_tongue_color(analysis),  # Custom extraction function
                "coating_color": extract_coating_color(analysis),
                "tongue_shape": extract_tongue_shape(analysis),
                "pattern_differentiation": extract_pattern(analysis)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {img_path.name}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Analysis completed. Results saved to {output_csv}")
    
    return df

# Helper functions to extract structured information
def extract_tongue_color(analysis_text):
    """Extract tongue body color"""
    color_keywords = {
        "pale white": "pale_white",
        "pale red": "pale_red", 
        "red": "red",
        "crimson": "crimson",
        "purple": "purple",
        "blue": "blue"
    }
    for keyword, color_code in color_keywords.items():
        if keyword.lower() in analysis_text.lower():
            return color_code
    return "unknown"

# Define other extraction functions similarly...
```

## 📊 Evaluation Benchmarks

We constructed specialized tongue diagnosis evaluation benchmarks covering multiple dimensions:

| Test Set | Samples | Description |
|----------|---------|-------------|
| **Tongue Body Color Test** | 1,200 images | Evaluate accuracy in identifying tongue body colors (pale white, pale red, red, crimson, purple, etc.) |
| **Tongue Coating Test** | 1,500 images | Evaluate identification of coating color (white, yellow, gray, black), thickness, moisture |
| **Tongue Shape Test** | 800 images | Evaluate recognition of tongue body shape (swollen, thin, scalloped, cracked, ecchymosis) |
| **Pattern Differentiation Test** | 1,000 images | Evaluate accuracy in eight-principle and organ system pattern differentiation |
| **Clinical Case Test** | 500 cases | Complete clinical cases with tongue images, evaluate clinical diagnostic capability |

### Evaluation Metrics

```python
def evaluate_tongue_model(model, test_dataset):
    """Evaluate tongue diagnosis model performance"""
    metrics = {
        "tongue_color_accuracy": 0.0,      # Tongue body color accuracy
        "coating_type_accuracy": 0.0,      # Tongue coating type accuracy
        "pattern_f1_score": 0.0,           # Pattern differentiation F1 score
        "clinical_accuracy": 0.0,          # Clinical diagnosis accuracy
        "consistency_score": 0.0           # Consistency with expert diagnosis
    }
    
    # Implement specific evaluation logic...
    return metrics
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch transformers accelerate pillow gradio
pip install opencv-python scikit-learn pandas  # Optional: for data processing
```

### 2. Basic Usage Example

```python
from tonguegpt import TongueAnalyzer

# Initialize analyzer
analyzer = TongueAnalyzer(model_name="TongueGPT-7B-VL")

# Analyze single tongue image
result = analyzer.analyze("path/to/tongue.jpg")
print(f"Diagnosis: {result['diagnosis']}")
print(f"Recommendations: {result['recommendations']}")

# Compare multiple tongue images (disease progression tracking)
comparison = analyzer.compare_images(
    ["day1.jpg", "day7.jpg", "day14.jpg"],
    descriptions=["Initial diagnosis", "After one week", "After two weeks"]
)
```

### 3. Integration into Existing Systems

```python
# Integrate TongueGPT into medical systems
class TCMDiagnosisSystem:
    def __init__(self):
        self.tongue_analyzer = TongueAnalyzer()
        self.symptom_analyzer = SymptomAnalyzer()  # Other symptom analysis modules
        
    def comprehensive_diagnosis(self, tongue_image, symptoms_text, pulse_data):
        """Comprehensive diagnosis integrating tongue, symptom, and pulse analysis"""
        # Tongue analysis
        tongue_result = self.tongue_analyzer.analyze(tongue_image)
        
        # Symptom analysis
        symptom_result = self.symptom_analyzer.analyze(symptoms_text)
        
        # Integrated judgment (add more logic as needed)
        final_diagnosis = self.integrate_diagnoses(
            tongue_result, symptom_result, pulse_data
        )
        
        return final_diagnosis
```

## 🏥 Clinical Applications

### 1. Clinical Decision Support
- Rapid tongue analysis in outpatient settings
- Pattern differentiation assistance
- Treatment efficacy evaluation

### 2. Health Management
- Daily tongue self-examination
- Sub-health status assessment
- Personalized health recommendations

### 3. Medical Education
- Tongue diagnosis teaching aid
- Case analysis and discussion
- Self-learning and assessment

### 4. Clinical Research
- Large-scale tongue image data analysis
- Pattern manifestation research
- Efficacy evaluation standard establishment

## 📖 Citation

If you use TongueGPT in your research, please cite:

```bibtex
@software{tonguegpt2025,
  title={TongueGPT: A Multimodal Large Language Model for Traditional Chinese Tongue Diagnosis},
}
```

## 🤝 Contributing

We welcome contributions in various forms:
- 🔍 Report issues or suggest improvements
- 📊 Contribute new tongue image data
- 💻 Submit code improvements
- 📖 Improve documentation and tutorials

## 📄 License

This project is licensed under [Apache License 2.0](LICENSE).

## 🏥 Disclaimer

**Important Notice**: This model serves only as an auxiliary diagnostic tool and cannot replace professional medical diagnosis. All diagnostic results and recommendations are for reference only. Actual diagnosis and treatment should be based on professional medical judgment. Consult qualified healthcare professionals before making any health decisions.

---

<div align="center">
<em>Empowering TCM Tongue Diagnosis with AI, Preserving Millennia of Wisdom</em>
</div>
