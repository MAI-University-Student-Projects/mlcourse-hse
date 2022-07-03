import numpy as np
from collections import Counter

# засунуть feature_types из конструктора в fit метод и сделать преобразование признаков там
# либо глянуть на причину его присутствия в цикле _fit_node

# в момент, когда формируется лист и дальше сплитить невозможно конкретно по одинаковости значений фичи - обыграть
# т.к. сейчас возвращается None, None, None, None в такой ситуации
class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, -np.inf, -np.inf, None #None,None,None,None
        for feature in range(0, sub_X.shape[1]): #1?
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = DecisionTree._find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best: #compare to None
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold #compare to None

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
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
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])
        
    @staticmethod
    def _find_best_split(feature_vector, target_vector):
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
        if (feature_vector == feature_vector[0]).all(): return None, None, None, None
        srt_vec = np.sort(feature_vector)
        # В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
        thrs = (srt_vec[:-1] + srt_vec[1:]) / 2
        # Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются
        if thrs[0] == srt_vec[0]: thrs = thrs[thrs > srt_vec[0]]
        if thrs[-1] == srt_vec[-1]: thrs = thrs[thrs < srt_vec[-1]]
        vec_tg = np.vstack((feature_vector, target_vector)).T
        p_m = (vec_tg[:, 1] == 1).sum() / vec_tg.shape[0]
        H_Rm = 2*p_m*(1 - p_m)
        
        def thrs_enum(t):
            mask = vec_tg[:, 0] < t
            Rl, Rr = vec_tg[mask], vec_tg[~mask]
            p_l, p_r = (Rl[:, 1] == 1).sum() / Rl.shape[0], (Rr[:, 1] == 1).sum() / Rr.shape[0]
            # $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
            H_Rl, H_Rr = 2*p_l*(1 - p_l), 2*p_r*(1 - p_r)
            # $$Q(R) = H(R) -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$
            Q_R = H_Rm - H_Rl*Rl.shape[0]/vec_tg.shape[0] - H_Rr*Rr.shape[0]/vec_tg.shape[0]
            return Q_R
        
        Q_R_ = np.array(list(map(thrs_enum, thrs)))
        return thrs, Q_R_, thrs[Q_R_.argmax()], Q_R_.max()
        
    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
