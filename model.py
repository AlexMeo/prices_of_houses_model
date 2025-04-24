import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


#Импорт данных в формате pandas Dataframe
train = pd.read_csv("train_prices.csv")
test = pd.read_csv("test_prices.csv")
#Сохранение столбца ID
train_ID = train['Id']
test_ID = test['Id']
#Удаление столбца ID из наших данных
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
plt.figure(figsize=(10, 6))
#Проверка распределения целевой переменной SalePrice
#Гистограмма с плотностью
ax = sns.histplot(train['SalePrice'], stat='density', alpha=0.5, kde=True)

#Нормальная кривая
mu, sigma = stats.norm.fit(train['SalePrice'])
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', lw=3, linestyle='--', label='Normal Fit')


#Настройки графика
plt.title('Distribution vs Normal Fit')
plt.xlabel('SalePrice')
plt.ylabel('Density')
plt.legend()
plt.show()

#Создание Q-Q plot
stats.probplot(train['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot для SalePrice')
plt.show()

#Легко заметить, что распределение не соответствует нормальному. Применим преобразование
train["SalePrice"] = np.log(train["SalePrice"])
# Создание Q-Q plot
stats.probplot(train['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot для SalePrice')
plt.show()

#Распределение близко к нормальному. Возможно использование линейной регрессии
#Нахождение выбросов по предиктору площадь
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# Удаление данных с большой площадью и низкой ценной
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index)

# Повторная проверка
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# Проверка отсутствующих значений переменных, очистка данных
y_train = train.SalePrice.values
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Проверка на отсутствие пропущенных значений
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())

#Изменение типов переменных
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#Перевод категориальных переменных в числовые
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data = pd.get_dummies(all_data)

#Преобразованные датасеты
train = all_data[:ntrain]
test = all_data[ntrain:]


#Метрика для расчета MSE в исходной шкале
def expm1_scorer(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return -mean_absolute_error(y_true, y_pred)


#Создаем кастомный скорер с обратным преобразованием
mse_scorer = make_scorer(expm1_scorer, greater_is_better=False)

#Инициализируем модели
models = {
    #Линейные модели
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),

    #Бустинговые алгоритмы
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1),

    #Стекинг модель
    'Stacking': StackingRegressor(
        estimators=[
            ('ridge', Ridge(alpha=1.0)),
            ('gb', GradientBoostingRegressor(n_estimators=100))
        ],
        final_estimator=LinearRegression()
    )
}

# Настройки кросс-валидации
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Сравнение моделей
for name, model in models.items():
    scores = cross_val_score(
        estimator=model,
        X=train,
        y=y_train,
        scoring=mse_scorer,
        cv=cv,
        n_jobs=-1
    )
    results[name] = {
        'mean_MAE': np.abs(np.mean(scores)),
        'std_MAE': np.std(scores)
    }

# Вывод результатов
print("{:<25} {:<20} {:<10}".format('Model', 'Mean MAE', 'Std MAE'))
print('-' * 50)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['mean_MAE']):
    print("{:<25} {:<20.4f} {:<10.4f}".format(
        name,
        metrics['mean_MAE'],
        metrics['std_MAE']
    ))

#Выбор лучшей модели
best_model_name = min(results.items(), key=lambda x: x[1]['mean_MAE'])[0]
best_model = models[best_model_name]
print(f"Лучшая модель: {best_model_name} с MAE = {results[best_model_name]['mean_MAE']:.2f}")

#Обучение на всех тренировочных данных
final_model = clone(best_model)
final_model.fit(train, y_train)
test_pred_log = final_model.predict(test)
test_pred = np.exp(test_pred_log)

#Сохранение в CSV
output = pd.DataFrame({
    'Id': test_ID,  # Если есть ID-столбец
    'Prediction': test_pred
})

# Сохранение без индекса
output.to_csv('submission.csv', index=False)