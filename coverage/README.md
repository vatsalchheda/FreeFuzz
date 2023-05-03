To run coverage, make sure you have py-cov, pytest, and coverage installed

#### Step 1: Creating Test Files
Create your test files by running ```create_test_files.py```. Mention your **FreeFuzz** Directory in the *Original_dir* variable. The test files and folder will be created in the location of ```create_test_files.py```

#### Step 2: Setup coverage files
Copy the ```.coveragerc``` file present in this folder and ```test/``` folder generated in the above step to your PaddlePaddle installation directory which will be something like this "C:\Users\user_id\AppData\Roaming\Python\Python39\site-packages\paddle\".

#### Step 3: Running commands
Open command prompt and go to PaddlePaddle installation directory.
Run this command \
```pytest --cov --cov-report=html:coverage-report```

HTML coverage report would be generate in the coverage-report folder. Open the index.html file to view the full report.
