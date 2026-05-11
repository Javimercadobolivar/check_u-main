# test/test_pipiline.py
import os
import time
from datetime import datetime
from ..core.orchestrator import OCRPipeline
from ..preprocess.image_cleaner import ImageCleaner
from ..preprocess.check_inclination import SkewDetector
from ..postprocess.cleaner import TextCleaner
from ..postprocess.structure import TextStructurer
from ..config import TEST_PDF_PATH

def test_complete_pipeline():
    """Test completo del pipeline OCR con preprocesamiento y postprocesamiento"""
    
    print("\n" + "="*60)
    print("TEST COMPLETO: OCR Pipeline con Pre/Postprocesamiento")
    print("="*60)
    
    # Verificar que el archivo de prueba existe
    if not os.path.exists(TEST_PDF_PATH):
        print(f"❌ Archivo de prueba no encontrado: {TEST_PDF_PATH}")
        return
    
    print(f"\n📄 Archivo de prueba: {TEST_PDF_PATH}")
    
    # 1. Test básico sin preprocesamiento
    print("\n" + "-"*60)
    print("1️⃣  EJECUTANDO: OCR básico (SIN preprocesamiento)")
    print("-"*60)
    
    try:
        pipeline = OCRPipeline()
        with open(TEST_PDF_PATH, "rb") as f:
            start_time = time.time()
            result = pipeline.process(f.read())
            elapsed = time.time() - start_time
        
        print(f"✓ OCR completado en {elapsed:.2f}s")
        print(f"✓ Caracteres extraídos: {len(result['text'])}")
        print("\n📝 Muestra de texto extraído (primeros 300 caracteres):")
        print(result['text'][:300])
    except Exception as e:
        print(f"❌ Error en OCR básico: {e}")
        return
    
    # 2. Test de preprocesamiento
    print("\n" + "-"*60)
    print("2️⃣  EJECUTANDO: Pruebas de preprocesamiento")
    print("-"*60)
    
    try:
        from pdf2image import convert_from_bytes
        with open(TEST_PDF_PATH, "rb") as f:
            images = convert_from_bytes(f.read())
        
        if images:
            img = images[0]
            print(f"✓ PDF convertido a {len(images)} imagen(es)")
            
            # Test Image Cleaner
            cleaner = ImageCleaner(resize_width=1200)
            import numpy as np
            img_array = np.array(img)
            cleaned_img = cleaner.clean_image(img)
            print(f"✓ Imagen limpiada: {cleaned_img.shape}")
            
            # Test Skew Detection
            skew_detector = SkewDetector()
            skew_angle = skew_detector.detect_skew(img_array)
            print(f"✓ Ángulo de inclinación detectado: {skew_angle:.2f}°")
            
            if abs(skew_angle) > 0.5:
                corrected_img, final_angle = skew_detector.auto_correct(img_array)
                print(f"✓ Imagen corregida: ángulo final {final_angle:.2f}°")
    except Exception as e:
        print(f"⚠️  Aviso en preprocesamiento: {e}")
    
    # 3. Test de postprocesamiento
    print("\n" + "-"*60)
    print("3️⃣  EJECUTANDO: Pruebas de postprocesamiento")
    print("-"*60)
    
    try:
        raw_text = result['text']
        
        # Test Text Cleaner
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(raw_text)
        print(f"✓ Texto limpiado")
        print(f"  - Antes: {len(raw_text)} caracteres")
        print(f"  - Después: {len(cleaned_text)} caracteres")
        print(f"  - Reducción: {((len(raw_text)-len(cleaned_text))/len(raw_text)*100):.1f}%")
        
        # Test Text Structurer
        structurer = TextStructurer()
        blocks = structurer.structure(cleaned_text)
        print(f"\n✓ Texto estructurado en {len(blocks)} bloques")
        
        # Contar tipos de bloques
        block_types = {}
        for block in blocks:
            block_types[block.type] = block_types.get(block.type, 0) + 1
        
        for btype, count in block_types.items():
            print(f"  - {btype}: {count}")
        
        # Test extracción de campos
        fields = structurer.extract_key_fields(cleaned_text)
        print(f"\n✓ Campos extraídos: {len(fields)}")
        for field, value in fields.items():
            if value:
                print(f"  - {field}: {value}")
    except Exception as e:
        print(f"❌ Error en postprocesamiento: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Resumen
    print("\n" + "="*60)
    print("✅ TEST COMPLETO FINALIZADO")
    print("="*60)
    print("\n📊 Estadísticas:")
    print(f"  - Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Archivo procesado: {os.path.basename(TEST_PDF_PATH)}")
    print(f"  - Tiempo total OCR: {elapsed:.2f}s")
    print(f"  - Tamaño de texto final: {len(cleaned_text)} caracteres")
    print("\n")

if __name__ == "__main__":
    test_complete_pipeline()
