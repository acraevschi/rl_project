import cv2
import numpy as np
import easyocr

def get_happiness_state(cropped_img):
    best_match = None
    best_val = 0
    threshold = 0.7
    states_dict = {"very_unhappy": 0.25, "unhappy": 0.5, "satisfied": 1, "happy": 1.25, "very_happy": 1.5}

    for state in states_dict.keys():
        template = cv2.imread(f"reward/happiness_levels/{state}.png", cv2.IMREAD_UNCHANGED)
        match_result = cv2.matchTemplate(cropped_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(match_result)
        if max_val > best_val:
            print(f"Match found with confidence: {max_val} for state: {state}")
            print(f"Best match so far was: {best_match}; changed to: {state}")
            best_val = max_val
            best_match = state

    return states_dict[best_match] if best_val >= threshold else None

get_happiness_state(cv2.imread("reward/scrns_test/Figure_5.png", cv2.IMREAD_UNCHANGED)) # may confuse the very_unhappy and unhappy states; and the very_happy and happy states

def get_population(image):
    """
    Get the population value and growth from the Cities Skylines population display.
    Args:
        image: OpenCV image in BGR format
    Returns:
        tuple: (population, growth) or None if failed
    """
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Basic preprocessing for better text detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Get text from the image
        results = reader.readtext(gray)
        
        # Process all detected text
        numbers = []
        for detection in results:
            text = detection[1].strip()
            
            # Try to identify if this is the population number (should contain a comma)
            if ',' in text:
                try:
                    # Remove any non-numeric characters except comma
                    clean_text = ''.join(c for c in text if c.isdigit() or c == ',')
                    population = int(clean_text.replace(',', ''))
                    numbers.append(population)
                except ValueError:
                    continue
            
            # Try to identify if this is the growth number (should start with + or -)
            elif text.startswith('+') or text.startswith('-'):
                try:
                    # Remove any non-numeric characters except + and -
                    clean_text = ''.join(c for c in text if c.isdigit() or c in '+-')
                    growth = int(clean_text)
                    numbers.append(growth)
                except ValueError:
                    continue
            
            # For cases where OCR might have missed the comma or +/-
            else:
                try:
                    # If it's a clean number, store it
                    num = int(''.join(c for c in text if c.isdigit() or c in '+-'))
                    numbers.append(num)
                except ValueError:
                    continue
        
        # Sort numbers by absolute value - population should be larger than growth
        numbers.sort(key=lambda x: abs(x), reverse=True)
        
        if len(numbers) >= 2:
            return (abs(numbers[0]), numbers[1])  # Population is always positive
        else:
            print(f"Found numbers: {numbers}")
            return None
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
    
# doesn't work perfectly, Figure_1 only returns the population, not the growth; Figure_4 return positive growth, but should be negative
get_population(cv2.imread("reward/scrns_test/Figure_4.png", cv2.IMREAD_UNCHANGED)) 

def extract_pop_happiness(cropped_img):
    """
    Extract the happiness value and the population from the cropped image.
    """

    scaler = get_happiness_state(cropped_img)
    ### get the population value

    return None

def extract_economy(cropped_img):
    """
    Extract the curernt finances and the growth rate from the cropped image.
    """
    return None

def extract_other_info(cropped_img):
    """
    Extract other information from the cropped image; will only be sporadically used.
    """
    return None