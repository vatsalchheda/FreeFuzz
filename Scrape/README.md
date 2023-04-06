# Scraping from web to obtain data for PaddlePaddle Instrumentation

1. Run the script ```scrape.py``` to obtain the list of APIs of PaddlePaddle.
2. To get api definitions and sampel code, first, replace "E:/UIUC/Spring 2023/CS 527/FreeFuzz/Scrape/chromedriver.exe" with the location of your Selenium Chrome Driver in ```scrape_api_def_documentation.py```
3. Run the python script mentioned in the point above. You will have two outputs: ```api_def.txt``` file which will have all the api definitions and ```api_documentation_code``` folder which will contain sample code for the APIs.
4. Run the script ```run_documentation.py``` to instrument PaddlePaddle library based on the code collected in above point.
