# Analiza Sentymentu LinkedIn
## Przegląd Projektu
Projekt przeprowadza analizę sentymentu opinii publicznych na temat OpenAI udostępnianych na platformie LinkedIn. Wykorzystując techniki przetwarzania języka naturalnego (NLP), system analizuje posty i komentarze, kategoryzując sentyment jako pozytywny, neutralny lub negatywny. Projekt został opracowany jako część pracy inżynierskiej.

## Funkcje
- Zbieranie danych z LinkedIn przy użyciu scrapera Apify
- Zaawansowane przetwarzanie wstępne tekstu obsługujące URL-e, slang, treści wielojęzyczne
- Funkcjonalność ręcznego etykietowania do tworzenia zbiorów danych treningowych
- Transfer Learning z modelem RoBERTa do klasyfikacji sentymentu
- Walidacja krzyżowa do oceny wydajności modelu
- Wizualizacja danych za pomocą histogramów i chmur słów

## Architektura
Projekt podąża za kompletnym procesem uczenia maszynowego:
1. **Zbieranie Danych**: Scrapowanie postów i komentarzy LinkedIn zawierających "openai"
2. **Przetwarzanie Wstępne**: Czyszczenie tekstu, wykrywanie języka, tokenizacja URL
3. **Etykietowanie**: Interfejs do ręcznego oznaczania sentymentu
4. **Trenowanie Modelu**: Dostrajanie modelu RoBERTa z wykorzystaniem Transfer Learning
5. **Ewaluacja**: Wykorzystanie walidacji krzyżowej i metryk dokładności
6. **Wizualizacja**: Wyświetlanie rozkładów sentymentu i częstotliwości słów

## Przetwarzanie Danych
- Wykrywanie języka angielskiego za pomocą biblioteki Lingua
- Standaryzacja URL i usuwanie duplikatów
- Kategoryzacja sentymentu (Pozytywny, Neutralny, Negatywny)

## Modele
Analiza sentymentu wykorzystuje:
- Model bazowy: RoBERTa
- Transfer Learning: Dostrajanie na danych specyficznych dla domeny
- Zamrażanie warstw: 10 dolnych warstw zamrożonych dla zachowania ogólnego zrozumienia języka

## Eksplorowane Koncepcje
Projekt eksploruje kilka kluczowych koncepcji nauki o danych i NLP:
- **Web Scraping**: Etyczne techniki zbierania danych z platform mediów społecznościowych
- **Przetwarzanie Języka Naturalnego**: Przetwarzanie wstępne tekstu, tokenizacja i analiza sentymentu
- **Transfer Learning**: Adaptacja wstępnie wytrenowanych modeli językowych do nowych domen
- **Inżynieria Cech**: Ekstrakcja istotnych cech z danych tekstowych
- **Ewaluacja Modelu**: Techniki walidacji krzyżowej i metryki wydajności
- **Nierównowaga Klas**: Obsługa nierównomiernego rozkładu klas sentymentu
- **Strojenie Hiperparametrów**: Optymalizacja parametrów modelu dla lepszej wydajności
- **Wizualizacja Danych**: Techniki prezentacji wyników analizy danych tekstowych

## Wymagania
- Python 3.10
- PyTorch
- Transformers (Hugging Face)
- pandas
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- Apify client
- Lingua

## Instalacja
```
pip install -r requirements.txt
```
Dodatkowo, projekt wymaga pakietu pytorch z wersją Nvidia CUDA kompatybilną z GPU, która może się różnić w zależności od urządzenia. Sprawdź wersję CUDA swojego GPU za pomocą: `nvidia-smi` i zainstaluj [kompatybilny pakiet pytorch](https://pytorch.org/).
