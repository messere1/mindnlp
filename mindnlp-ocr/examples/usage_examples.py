"""
VLM-OCR Engine - Example Usage

This file demonstrates various usage scenarios of the VLM-OCR engine.
"""

import os
from pathlib import Path
from core.engine import VLMOCREngine
from api.schemas.request import OCRRequest

# =============================================================================
# Example 1: Basic Single Image OCR
# =============================================================================

def example_basic_ocr():
    """Basic single image OCR with default settings."""
    print("\n" + "="*80)
    print("Example 1: Basic Single Image OCR")
    print("="*80)
    
    # Initialize engine
    engine = VLMOCREngine(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda"  # or "cpu"
    )
    
    # Load image
    image_path = "sample_images/chinese_text.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Create request
    request = OCRRequest(
        image=image_bytes,
        output_format="json",
        language="zh"
    )
    
    # Perform OCR
    response = engine.predict(request)
    
    # Print results
    print(f"\nSuccess: {response.success}")
    print(f"Texts: {response.texts}")
    print(f"Boxes: {response.boxes}")
    print(f"Confidences: {response.confidences}")
    print(f"Inference time: {response.inference_time:.3f}s")


# =============================================================================
# Example 2: Document Understanding with Markdown Output
# =============================================================================

def example_document_understanding():
    """Document understanding with structured Markdown output."""
    print("\n" + "="*80)
    print("Example 2: Document Understanding")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/document.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    request = OCRRequest(
        image=image_bytes,
        output_format="markdown",  # Structured output
        language="zh",
        task_type="document"       # Document understanding mode
    )
    
    response = engine.predict(request)
    
    print(f"\nDocument structure:\n{response.texts[0]}")
    print(f"Inference time: {response.inference_time:.3f}s")


# =============================================================================
# Example 3: Table Recognition
# =============================================================================

def example_table_recognition():
    """Table recognition with JSON output."""
    print("\n" + "="*80)
    print("Example 3: Table Recognition")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/table.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    request = OCRRequest(
        image=image_bytes,
        output_format="json",
        language="zh",
        task_type="table"  # Table recognition mode
    )
    
    response = engine.predict(request)
    
    print("\nTable data:")
    for i, (text, box) in enumerate(zip(response.texts, response.boxes)):
        print(f"  Cell {i+1}: {text}")
        print(f"    Position: {box}")
    print(f"Inference time: {response.inference_time:.3f}s")


# =============================================================================
# Example 4: Formula Recognition
# =============================================================================

def example_formula_recognition():
    """Mathematical formula recognition."""
    print("\n" + "="*80)
    print("Example 4: Formula Recognition")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/formula.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    request = OCRRequest(
        image=image_bytes,
        output_format="text",
        language="en",
        task_type="formula"  # Formula recognition mode
    )
    
    response = engine.predict(request)
    
    print(f"\nFormula (LaTeX): {response.texts[0]}")
    print(f"Inference time: {response.inference_time:.3f}s")


# =============================================================================
# Example 5: Batch Processing
# =============================================================================

def example_batch_processing():
    """Batch processing multiple images."""
    print("\n" + "="*80)
    print("Example 5: Batch Processing")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    # Load multiple images
    image_dir = Path("sample_images")
    image_paths = list(image_dir.glob("*.jpg"))[:5]  # First 5 images
    
    # Create batch request
    requests = []
    for path in image_paths:
        with open(path, "rb") as f:
            requests.append(OCRRequest(
                image=f.read(),
                output_format="json",
                language="zh"
            ))
    
    # Batch processing
    print(f"\nProcessing {len(requests)} images...")
    responses = engine.predict_batch(requests)
    
    # Print results
    total_time = 0
    for i, response in enumerate(responses):
        print(f"\nImage {i+1} ({image_paths[i].name}):")
        print(f"  Texts: {response.texts}")
        print(f"  Time: {response.inference_time:.3f}s")
        total_time += response.inference_time
    
    avg_time = total_time / len(responses)
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Average time per image: {avg_time:.3f}s")


# =============================================================================
# Example 6: URL Image OCR
# =============================================================================

def example_url_ocr():
    """OCR from image URL."""
    print("\n" + "="*80)
    print("Example 6: URL Image OCR")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    # Image URL
    image_url = "https://example.com/sample_image.jpg"
    
    request = OCRRequest(
        image_url=image_url,
        output_format="json",
        language="en"
    )
    
    response = engine.predict_from_url(request)
    
    print(f"\nURL: {image_url}")
    print(f"Texts: {response.texts}")
    print(f"Inference time: {response.inference_time:.3f}s")


# =============================================================================
# Example 7: Custom Prompt
# =============================================================================

def example_custom_prompt():
    """Using custom prompt for specific extraction."""
    print("\n" + "="*80)
    print("Example 7: Custom Prompt")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/business_card.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Custom prompt for business card extraction
    custom_prompt = """
    Please extract the following information from this business card:
    - Name
    - Title
    - Company
    - Phone
    - Email
    
    Return in JSON format.
    """
    
    request = OCRRequest(
        image=image_bytes,
        output_format="json",
        language="en",
        custom_prompt=custom_prompt
    )
    
    response = engine.predict(request)
    
    print(f"\nExtracted information:")
    print(response.texts[0])


# =============================================================================
# Example 8: Multi-language OCR
# =============================================================================

def example_multilingual_ocr():
    """Multi-language text recognition."""
    print("\n" + "="*80)
    print("Example 8: Multi-language OCR")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/multilingual.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    request = OCRRequest(
        image=image_bytes,
        output_format="json",
        language="multi"  # Multi-language mode
    )
    
    response = engine.predict(request)
    
    print("\nDetected texts:")
    for i, text in enumerate(response.texts):
        print(f"  {i+1}. {text}")


# =============================================================================
# Example 9: High Confidence Filtering
# =============================================================================

def example_confidence_filtering():
    """Filter results by confidence threshold."""
    print("\n" + "="*80)
    print("Example 9: Confidence Filtering")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/noisy_image.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # High confidence threshold for noisy images
    request = OCRRequest(
        image=image_bytes,
        output_format="json",
        language="zh",
        confidence_threshold=0.9  # Only high-confidence results
    )
    
    response = engine.predict(request)
    
    print("\nHigh-confidence results:")
    for text, conf in zip(response.texts, response.confidences):
        print(f"  {text} (confidence: {conf:.2f})")


# =============================================================================
# Example 10: Different Output Formats
# =============================================================================

def example_output_formats():
    """Compare different output formats."""
    print("\n" + "="*80)
    print("Example 10: Different Output Formats")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    image_path = "sample_images/mixed_content.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    formats = ["json", "text", "markdown"]
    
    for fmt in formats:
        print(f"\n--- {fmt.upper()} Format ---")
        
        request = OCRRequest(
            image=image_bytes,
            output_format=fmt,
            language="zh"
        )
        
        response = engine.predict(request)
        
        if fmt == "json":
            print(f"Texts: {response.texts}")
            print(f"Boxes: {response.boxes}")
            print(f"Confidences: {response.confidences}")
        else:
            print(response.texts[0])


# =============================================================================
# Example 11: Error Handling
# =============================================================================

def example_error_handling():
    """Demonstrate error handling."""
    print("\n" + "="*80)
    print("Example 11: Error Handling")
    print("="*80)
    
    engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    # Test 1: Invalid image format
    print("\nTest 1: Invalid image format")
    try:
        request = OCRRequest(
            image=b"not an image",
            output_format="json",
            language="zh"
        )
        response = engine.predict(request)
    except Exception as e:
        print(f"  Error caught: {type(e).__name__}: {e}")
    
    # Test 2: Image too large
    print("\nTest 2: Image too large")
    try:
        # Create a large dummy image (> 10MB)
        large_image = b"X" * (11 * 1024 * 1024)
        request = OCRRequest(
            image=large_image,
            output_format="json",
            language="zh"
        )
        response = engine.predict(request)
    except Exception as e:
        print(f"  Error caught: {type(e).__name__}: {e}")
    
    # Test 3: Invalid parameters
    print("\nTest 3: Invalid parameters")
    try:
        with open("sample_images/test.jpg", "rb") as f:
            request = OCRRequest(
                image=f.read(),
                output_format="invalid",  # Invalid format
                language="zh"
            )
            response = engine.predict(request)
    except Exception as e:
        print(f"  Error caught: {type(e).__name__}: {e}")


# =============================================================================
# Example 12: Performance Benchmark
# =============================================================================

def example_performance_benchmark():
    """Benchmark different configurations."""
    print("\n" + "="*80)
    print("Example 12: Performance Benchmark")
    print("="*80)
    
    import time
    
    # Test configurations
    configs = [
        {"device": "cuda", "batch_size": 1},
        {"device": "cuda", "batch_size": 4},
        {"device": "cuda", "batch_size": 8},
    ]
    
    image_path = "sample_images/test.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    for config in configs:
        print(f"\nConfig: {config}")
        
        engine = VLMOCREngine(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            device=config["device"]
        )
        
        # Warm-up
        request = OCRRequest(image=image_bytes, output_format="json", language="zh")
        engine.predict(request)
        
        # Benchmark
        num_iterations = 10
        start = time.time()
        
        for _ in range(num_iterations):
            response = engine.predict(request)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_iterations
        
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Throughput: {1/avg_time:.2f} images/s")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("="*80)
    print("VLM-OCR Engine - Example Usage")
    print("="*80)
    
    examples = [
        ("Basic OCR", example_basic_ocr),
        ("Document Understanding", example_document_understanding),
        ("Table Recognition", example_table_recognition),
        ("Formula Recognition", example_formula_recognition),
        ("Batch Processing", example_batch_processing),
        ("URL OCR", example_url_ocr),
        ("Custom Prompt", example_custom_prompt),
        ("Multi-language OCR", example_multilingual_ocr),
        ("Confidence Filtering", example_confidence_filtering),
        ("Output Formats", example_output_formats),
        ("Error Handling", example_error_handling),
        ("Performance Benchmark", example_performance_benchmark),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun specific example: python examples.py <number>")
    print("Run all examples: python examples.py all")
    
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "all":
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\nExample '{name}' failed: {e}")
        else:
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(examples):
                    name, func = examples[idx]
                    func()
                else:
                    print(f"Invalid example number: {arg}")
            except ValueError:
                print(f"Invalid argument: {arg}")
    else:
        print("\nNo example specified. Use 'python examples.py <number>' or 'python examples.py all'")


if __name__ == "__main__":
    main()
