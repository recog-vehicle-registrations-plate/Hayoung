# Hayoung

    1) ESR-GAN
       clone from "https://github.com/xinntao/ESRGAN"
       Follow README.md step in ESRGAN github.
       
    2) 자동차 번호판 인식기.ipynb
    
    3) 7기/image_histogram/image_histo.py
        Add morphological operation(erosion)
        
    4) 7기 분석 결과
        모든 테스트 이미지를 한줄짜리 번호판을 넣어두고, 테스트를 한 것으로 보임.
        Window Sliding 을 적용하여 글자 하나씩 추출해서 OCR-tesseract를 사용하지 않고 
        별도의 모델을 생성해서 문제를 해결하려고 한 것 같음.   