# Summary of the project
This project was for the purposes of Privacy Engineering in Spring 2023 taught by Dr. Hana Habib.

In this project, we analyzed a location dataset and how we can preserve privacy using Differential Privacy. Specifically, we used NYC’s Taxi Trip Data set that was released in 2013. The dataset includes trip records of all yellow taxis in NYC in 2013, the records includes fields such as pick-up and drop-off locations, trip distances and payment types. The data was made available due to the Taxicab Passenger Enhancement Program (TPEP) initiative.

We used IBM’s Differential Privacy library to analyze the dataset so that we can utilize it while preserving users' privacy. The goal of the project is to understand the different mechanisms of differential privacy and apply it to our dataset. We applied the Laplace and Guassian mechanism to our dataset and compared the performance with a non-private baseline (no noise added to the dataset) and wanted to predict trip distance, trip duration and passenger count. We were able to achieve this goal with high accuracy.

The importance of the project is that data has become an invaluable asset for companies across all industries. By employing Machine Learning techniques, companies collect vast amounts of data from users to better understand their behavior and offer personalized experiences. However, careless data gathering can lead to a serious breach of privacy. It is now more important than ever to prioritize privacy when using ML models. Previous privacy measures, such as k-anonymity, have been implemented, but in 2006, Cynthia Dwork and her colleagues developed a new method called Differential Privacy which is the de facto method to achieve privacy and utility in industry. This is particularly relevant for location tracking datasets, as it has implications for services like Uber and DoorDash that rely heavily on location data.

The project consists of three major parts. First, a non-private baseline is implemented, which does not apply differential privacy to the dataset. This serves as a reference for evaluating the performance of the differentially private models. Second, we import and apply the DP linear regression model in IBM's diffpriv library. Finally, we made a customized version of the linear regression model by incorporating IBM's Gaussian and Laplace mechanisms. Overall, the project aims to explore and compare different approaches to achieve privacy-preserving machine learning.


# How to run the code ?

There is no specific order to run the python scripts, but for regular linear regression run linear_regression.py. For applying the laplace or gaussain mechanism use the customized_linear_regression.py and change line 21 to laplace or guassian. For comparing results between different mechanisms run the compare_reuslt.py


# Group members - Team 5
Faisal Binmahfoudh (faisalb), Noora Alfayez (nalfayez), Wenhao Song (wenhao2) and Yujie Wang (yujiewan)
