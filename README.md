# preprocessing_pipeline_chicago

Creates a package for preprocessing a specific dataset with mixed data.

Tests the process by predicting classes with a random forest classifier, saving the model, and using the pickled version on new data.

## Packaged functions

1. column_separator is given a dataframe, a target variable, and identifiers for special columns with either dates or police beats. The function then identifies variables as either numeric or categorical, excluding the special and target columns.

2. sample_splitter is given a dataframe, a target variable, a subsample value, and a test size, and splits the data into X and y, train and test.

## Packaged transformers

1. NullMaker is given a dataframe and looks for string values that should be represented by nulls, such as "UNKNOWN", then converts them to np.nan.

2. BeatFormatter is given the beat of occurence column and converts the beat number to a rolled-up two digit value.

3. DateFormatter is given columns of dates represented as strings and converts them to a pandas Timestamp.

4. DateEncoder is given columns of Timestamp data and extracts the year, month, day, weekday, and hour, making them into new columns.

## Model training

* pipeline_preprocessing_forest creates a modeling pipeline from data ingestion to validation and prediction, then pickles the trained model.

## Model usage

* pickle_use shows an example of usage, loading the pickled model and using it to predict on new data.
