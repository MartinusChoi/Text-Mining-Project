library("reticulate")
py_install("pandas")

py_install("pickle")

source_python("read_pickle.py")
pickle_data <- read_pickle_file("lemmatized-all.pkl")

pickle_data