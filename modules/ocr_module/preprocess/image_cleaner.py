# preprocess/image_cleaner.py
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

class ImageCleaner:
    """Limpia y normaliza imágenes para mejorar OCR"""
    
    def __init__(self, resize_width: int = 1200, blur_kernel: Tuple[int, int] = (5, 5)):
        self.resize_width = resize_width
        self.blur_kernel = blur_kernel
    
    def clean_image(self, image) -> np.ndarray:
        """
        Limpia una imagen aplicando varias técnicas
        
        Args:
            image: Imagen PIL o array numpy
        
        Returns:
            Imagen limpia como array numpy
        """
        # Convertir a numpy array si es PIL
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convertir a escala de grises si es color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Redimensionar
        img = self._resize_image(img)
        
        # Desenfoque gaussiano
        img = cv2.GaussianBlur(img, self.blur_kernel, 0)
        
        # Binarización adaptativa
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Eliminar ruido
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        return img
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Redimensiona la imagen preservando la relación de aspecto"""
        height, width = image.shape[:2]
        if width > self.resize_width:
            scale = self.resize_width / width
            new_height = int(height * scale)
            image = cv2.resize(image, (self.resize_width, new_height))
        return image
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
        """
        Mejora el contraste de la imagen
        
        Args:
            image: Imagen de entrada
            alpha: Factor de contraste (>1 aumenta contraste)
            beta: Brillo
        
        Returns:
            Imagen con contraste mejorado
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Intenta eliminar sombras de la imagen"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        return opening

def clean_image(image, resize_width: int = 1200) -> np.ndarray:
    """Función de conveniencia para limpiar una imagen"""
    cleaner = ImageCleaner(resize_width=resize_width)
    return cleaner.clean_image(image)
