import numpy as np

class SGDRegressor:
    def __init__(self, alpha=0.01, learning_rate=0.01, max_iter=1000, tol=1e-4, random_state=None):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0.0

        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for idx in indices:
                xi = X[idx:idx+1]
                yi = y[idx:idx+1]

                prediction = np.dot(xi, self.coef_) + self.intercept_
                error = prediction - yi

                grad_w = xi.T.dot(error) + self.alpha * self.coef_
                grad_b = error

                self.coef_ -= self.learning_rate * grad_w.flatten()
                self.intercept_ -= self.learning_rate * grad_b.flatten()[0]

            y_pred = self.predict(X)
            loss = np.mean((y - y_pred) ** 2) + 0.5 * self.alpha * np.sum(self.coef_ ** 2)

            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                break

            prev_loss = loss
            self.n_iter_ = iteration + 1

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        A = X_centered.T.dot(X_centered) + self.alpha * np.eye(n_features)
        b = X_centered.T.dot(y_centered)

        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class LassoRegressor:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def _soft_threshold(self, x, lambda_):
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0.0

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        self.coef_ = np.zeros(n_features)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                X_j = X_centered[:, j]
                y_pred_without_j = np.dot(X_centered, self.coef_) - X_j * self.coef_[j]
                rho_j = np.dot(X_j, y_centered - y_pred_without_j)
                z_j = np.dot(X_j, X_j)

                if z_j > 0:
                    self.coef_[j] = self._soft_threshold(rho_j / z_j, self.alpha / z_j)
                else:
                    self.coef_[j] = 0.0

            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

            self.n_iter_ = iteration + 1

        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class ElasticNetRegressor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def _soft_threshold(self, x, lambda_):
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0.0

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        self.coef_ = np.zeros(n_features)

        lambda1 = self.alpha * self.l1_ratio
        lambda2 = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                X_j = X_centered[:, j]
                y_pred_without_j = np.dot(X_centered, self.coef_) - X_j * self.coef_[j]
                rho_j = np.dot(X_j, y_centered - y_pred_without_j)
                z_j = np.dot(X_j, X_j) + lambda2

                if z_j > 0:
                    self.coef_[j] = self._soft_threshold(rho_j / z_j, lambda1 / z_j)
                else:
                    self.coef_[j] = 0.0

            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

            self.n_iter_ = iteration + 1

        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features

        n_output_features = 0
        if self.include_bias:
            n_output_features += 1

        n_output_features += n_features

        if self.degree >= 2:
            n_output_features += n_features * (n_features + 1) // 2

        self.n_output_features_ = n_output_features
        return self

    def transform(self, X):
        n_samples, n_features = X.shape

        features = []

        if self.include_bias:
            features.append(np.ones((n_samples, 1)))

        features.append(X)

        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(features)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MSE': mse
    }


class GridSearchCV:
    def __init__(self, model_class, param_grid, cv=5, scoring='rmse', random_state=None):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = []

    def _get_param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        self._generate_combinations(keys, values, 0, {}, combinations)
        return combinations

    def _generate_combinations(self, keys, values, index, current, combinations):
        if index == len(keys):
            combinations.append(current.copy())
            return

        key = keys[index]
        for value in values[index]:
            current[key] = value
            self._generate_combinations(keys, values, index + 1, current, combinations)

    def fit(self, X, y):
        param_combinations = self._get_param_combinations()

        n_samples = len(X)
        fold_size = n_samples // self.cv

        best_score = float('inf')
        best_params = None

        for params in param_combinations:
            fold_scores = []

            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < self.cv - 1 else n_samples

                train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
                val_indices = list(range(val_start, val_end))

                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]

                model = self.model_class(**params)
                if self.random_state is not None and 'random_state' in params:
                    model.random_state = self.random_state

                model.fit(X_train_fold, y_train_fold)
                y_pred_fold = model.predict(X_val_fold)

                if self.scoring == 'rmse':
                    score = np.sqrt(np.mean((y_val_fold - y_pred_fold) ** 2))
                elif self.scoring == 'mae':
                    score = np.mean(np.abs(y_val_fold - y_pred_fold))
                else:
                    score = np.mean((y_val_fold - y_pred_fold) ** 2)

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            self.cv_results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': np.std(fold_scores)
            })

            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        self.best_estimator_ = self.model_class(**best_params)
        if self.random_state is not None and 'random_state' in best_params:
            self.best_estimator_.random_state = self.random_state
        self.best_estimator_.fit(X, y)

        return self
