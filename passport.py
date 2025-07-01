import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from datetime import datetime
import os
import json

class IndianPassportOCR:
    def __init__(self, tesseract_path=None):
        """
        Initialize the Indian Passport OCR class
        
        Args:
            tesseract_path (str): Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Configure Tesseract for better OCR results
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/()<>-. '
    
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding for better text clarity
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Extracted text
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(processed_image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def extract_name(self, text):
        """
        Extract name from passport text
        
        Args:
            text (str): OCR extracted text
            
        Returns:
            dict: Dictionary containing surname and given name
        """
        name_info = {"surname": "", "given_name": ""}
        
        # Pattern to match surname (after "Surname" or "उपनाम")
        surname_patterns = [
            r'(?:Surname|उपनाम)[:\s]*([A-Z\s]+)',
            r'उपनाम[:\s]*([A-Z\s]+)',
            r'SURNAME[:\s]*([A-Z\s]+)'
        ]
        
        # Pattern to match given name (after "Given Name" or "दिया गया नाम")
        given_name_patterns = [
            r'(?:Given Name|दिया गया नाम)[:\s]*([A-Z\s]+)',
            r'Given Name\(s\)[:\s]*([A-Z\s]+)',
            r'GIVEN NAME\(S\)[:\s]*([A-Z\s]+)'
        ]
        
        # Extract surname
        for pattern in surname_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                surname = match.group(1).strip()
                # Clean up the surname (remove extra spaces, numbers)
                surname = re.sub(r'\s+', ' ', surname)
                surname = re.sub(r'[0-9/]', '', surname).strip()
                if len(surname) > 1:
                    name_info["surname"] = surname
                    break
        
        # Extract given name
        for pattern in given_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                given_name = match.group(1).strip()
                # Clean up the given name
                given_name = re.sub(r'\s+', ' ', given_name)
                given_name = re.sub(r'[0-9/]', '', given_name).strip()
                if len(given_name) > 1:
                    name_info["given_name"] = given_name
                    break
        
        return name_info
    
    def extract_visa_validity(self, text):
        """
        Extract visa validity period from passport text
        
        Args:
            text (str): OCR extracted text
            
        Returns:
            dict: Dictionary containing visa validity information
        """
        visa_info = {"issue_date": "", "expiry_date": "", "validity_period": ""}
        
        # Pattern to match dates in DD/MM/YYYY format
        date_patterns = [
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}-\d{2}-\d{4})',
            r'(\d{2}\.\d{2}\.\d{4})'
        ]
        
        # Pattern to match issue and expiry dates
        issue_patterns = [
            r'(?:Date of Issue|जारी करने की तिथि)[:\s]*(\d{2}/\d{2}/\d{4})',
            r'(?:Issue Date|जारी तिथि)[:\s]*(\d{2}/\d{2}/\d{4})',
            r'DATE OF ISSUE[:\s]*(\d{2}/\d{2}/\d{4})'
        ]
        
        expiry_patterns = [
            r'(?:Date of Expiry|समाप्ति की तिथि)[:\s]*(\d{2}/\d{2}/\d{4})',
            r'(?:Expiry Date|समाप्ति तिथि)[:\s]*(\d{2}/\d{2}/\d{4})',
            r'DATE OF EXPIRY[:\s]*(\d{2}/\d{2}/\d{4})'
        ]
        
        # Extract issue date
        for pattern in issue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                visa_info["issue_date"] = match.group(1)
                break
        
        # Extract expiry date
        for pattern in expiry_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                visa_info["expiry_date"] = match.group(1)
                break
        
        # If specific patterns don't work, try to find all dates and infer
        if not visa_info["issue_date"] or not visa_info["expiry_date"]:
            all_dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                all_dates.extend(matches)
            
            # Sort dates and try to identify issue and expiry
            if len(all_dates) >= 2:
                try:
                    # Convert to datetime objects for comparison
                    date_objects = []
                    for date_str in all_dates:
                        try:
                            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                            date_objects.append((date_str, date_obj))
                        except ValueError:
                            continue
                    
                    # Sort by date
                    date_objects.sort(key=lambda x: x[1])
                    
                    if len(date_objects) >= 2:
                        if not visa_info["issue_date"]:
                            visa_info["issue_date"] = date_objects[0][0]
                        if not visa_info["expiry_date"]:
                            visa_info["expiry_date"] = date_objects[-1][0]
                
                except Exception as e:
                    print(f"Error processing dates: {e}")
        
        # Calculate validity period
        if visa_info["issue_date"] and visa_info["expiry_date"]:
            try:
                issue_date = datetime.strptime(visa_info["issue_date"], '%d/%m/%Y')
                expiry_date = datetime.strptime(visa_info["expiry_date"], '%d/%m/%Y')
                validity_days = (expiry_date - issue_date).days
                visa_info["validity_period"] = f"{validity_days} days"
            except ValueError:
                visa_info["validity_period"] = "Unable to calculate"
        
        return visa_info
    
    def process_passport(self, file_path):
        """
        Process passport file and extract required information
        
        Args:
            file_path (str): Path to passport file (PDF or image)
            
        Returns:
            dict: Dictionary containing extracted information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and extract text
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            text = self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Extract information
        name_info = self.extract_name(text)
        visa_info = self.extract_visa_validity(text)
        
        # Combine results
        result = {
            "file_path": file_path,
            "extraction_timestamp": datetime.now().isoformat(),
            "name": name_info,
            "visa_validity": visa_info,
            "raw_text": text
        }
        
        return result
    
    def save_results(self, results, output_path):
        """
        Save extraction results to JSON file
        
        Args:
            results (dict): Extraction results
            output_path (str): Path to save JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """
    Main function to demonstrate usage
    """
    # Initialize OCR processor
    ocr = IndianPassportOCR()
    
    # Example usage
    try:
        # Replace with your passport file path
        passport_file = "/home/stark/Documents/passport clone/Text-Extraction-From-Image/page_1.png"  # or .jpg, .png, etc.
        
        # Process passport
        print("Processing passport...")
        results = ocr.process_passport(passport_file)
        
        # Display results
        print("\n" + "="*50)
        print("PASSPORT EXTRACTION RESULTS")
        print("="*50)
        
        print(f"\nName Information:")
        print(f"  Surname: {results['name']['surname']}")
        print(f"  Given Name: {results['name']['given_name']}")
        
        print(f"\nVisa Validity Information:")
        print(f"  Issue Date: {results['visa_validity']['issue_date']}")
        print(f"  Expiry Date: {results['visa_validity']['expiry_date']}")
        print(f"  Validity Period: {results['visa_validity']['validity_period']}")
        
        # Save results
        output_file = "passport_extraction_results.json"
        ocr.save_results(results, output_file)
        
    except Exception as e:
        print(f"Error processing passport: {e}")

if __name__ == "__main__":
    main()
