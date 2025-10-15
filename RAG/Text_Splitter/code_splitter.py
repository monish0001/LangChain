from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=280, chunk_overlap=0,
    language=Language.PYTHON
)

text = """

    # ai_class.py
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    class AIModel:
  

        def __init__(self, model=None):

            self.model = model if model else RandomForestClassifier()
            self.is_trained = False

        def train(self, X, y, test_size=0.2, random_state=42):
 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return X_test, y_test

        def predict(self, X):
 
            if not self.is_trained:
                raise ValueError("Model is not trained yet!")
            return self.model.predict(X)

        def evaluate(self, X_test, y_test):

            predictions = self.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            return accuracy


    # Example usage
    if __name__ == "__main__":
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target

        ai = AIModel()  # Initialize AI class
        X_test, y_test = ai.train(X, y)
        accuracy = ai.evaluate(X_test, y_test)
        print(f"Model Accuracy: {accuracy:.2f}")
        
"""

cunks = text_splitter.split_text(text)
print(f"Number of chunks: {len(cunks)}")
print(cunks[0])



# https://chunkviz.up.railway.app/