import numpy as np
from collections import Counter

# засунуть feature_types из конструктора в fit метод и сделать преобразование признаков там
# либо глянуть на причину его присутствия в цикле _fit_node

# в момент, когда формируется лист и дальше сплитить невозможно конкретно по одинаковости значений фичи - обыграть
# т.к. сейчас возвращается None, None, None, None в такой ситуации
class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=3):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        
        self._max_depth = np.inf if max_depth is None else max_depth
        self._depth = -np.inf
        
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self._depth = depth if depth > self._depth else self._depth
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        if (sub_X.shape[0] > self._min_samples_leaf) and (depth < self._max_depth):
            for feature in range(0, sub_X.shape[1]):
                feature_type = self._feature_types[feature]

                if feature_type == "real":
                    feature_vector = sub_X[:, feature]
                elif feature_type == "categorical":
                    counts = Counter(sub_X[:, feature])
                    clicks = Counter(sub_X[sub_y == self._pos_label, feature])
                    ratio = dict((key, clicks[key] / current_count) if key in clicks else (key, 0) for key, current_count in counts.items())
                    feature_vector = np.array((list(map(lambda x: ratio[x], sub_X[:, feature]))))
                else:
                    raise ValueError

                _, _, threshold, gini = DecisionTree._find_best_split(feature_vector, sub_y, self._pos_label)
                if gini_best is None or (gini > gini_best):
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold

                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, ratio.items()))) #[]categories_map.items())
                    else:
                        raise ValueError

        if (feature_best is None) or (sub_X[split].shape[0] < self._min_samples_split) or (sub_X[~split].shape[0] < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            self._depth = depth if depth > self._depth else self._depth
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth+1)
        
    @staticmethod
    def _find_best_split(feature_vector, target_vector, pos_label=1):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        """
        $$Q(R) = H(R) -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
        $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево
        
        Под критерием Джини здесь подразумевается следующая функция:
         $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
        Указания:
        * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
        * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
        * Поведение функции в случае константного признака может быть любым.
        * При одинаковых приростах Джини нужно выбирать минимальный сплит.
        * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

        :param feature_vector: вещественнозначный вектор значений признака
        :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

        :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
         разделить на две различные подвыборки, или поддерева
        :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
        :return threshold_best: оптимальный порог (число)
        :return gini_best: оптимальное значение критерия Джини (число)
        """
        
        # Поведение функции в случае константного признака может быть любым

        if (feature_vector == feature_vector[0]).all(): return np.nan, np.nan, np.nan, np.nan
        srt_vec = np.sort(feature_vector)
        # В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
        thrs = (srt_vec[:-1] + srt_vec[1:]) / 2
        # Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются
        if thrs[0] == srt_vec[0]: thrs = thrs[thrs > srt_vec[0]]
        if thrs[-1] == srt_vec[-1]: thrs = thrs[thrs < srt_vec[-1]]
        # vec_tg = np.vstack((feature_vector, target_vector)).T
        p_m = (target_vector == pos_label).sum() / target_vector.shape[0]
        H_Rm = 2*p_m*(1 - p_m)
        
        def thrs_enum(t):
            mask = feature_vector < t
            Rl, Rr = target_vector[mask], target_vector[~mask]
            p_l, p_r = (Rl == pos_label).sum() / Rl.shape[0], (Rr == pos_label).sum() / Rr.shape[0]
            # $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
            H_Rl, H_Rr = 2*p_l*(1 - p_l), 2*p_r*(1 - p_r)
            # $$Q(R) = H(R) -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$
            Q_R = H_Rm - H_Rl*Rl.shape[0]/target_vector.shape[0] - H_Rr*Rr.shape[0]/target_vector.shape[0]
            return Q_R
        
        Q_R_ = np.array(list(map(thrs_enum, thrs)))
        return thrs, Q_R_, thrs[Q_R_.argmax()], Q_R_.max()
        
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_to_split = node['feature_split']
        if self._feature_types[feature_to_split] == 'real':
            return self._predict_node(x, node["left_child"]) if x[feature_to_split] < node['threshold'] else self._predict_node(x, node["right_child"])
        else:
            return self._predict_node(x, node["left_child"]) if np.isin(x[feature_to_split], node['categories_split']) else self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._pos_label = np.unique(y)[-1]
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = [self._predict_node(x, self._tree) for x in X]
        return np.array(predicted)

    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }