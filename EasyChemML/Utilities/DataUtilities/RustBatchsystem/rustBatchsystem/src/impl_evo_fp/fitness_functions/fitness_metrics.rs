use itertools::enumerate;
use linfa::metrics::{SingleTargetRegression, ToConfusionMatrix};
use linfa::Dataset;
use ndarray::{Array1, Ix1};
use std::fmt::Display;

#[derive(Clone, Debug, PartialEq)]
pub enum MetricDirection {
    Maximize,
    Minimize,
    OneBest,
}

#[derive(Clone, Debug)]
pub enum Metric {
    ClassificationMetric(ClassificationMetric),
    RegressionMetric(RegressionMetric),
}

impl Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::ClassificationMetric(classification_metric) => {
                write!(f, "{:?}", classification_metric)
            }
            Metric::RegressionMetric(regression_metric) => write!(f, "{:?}", regression_metric),
        }
    }
}

impl Metric {
    pub fn direction(&self) -> MetricDirection {
        match self {
            Metric::ClassificationMetric(classification_metric) => {
                classification_metric.direction()
            }
            Metric::RegressionMetric(regression_metric) => regression_metric.direction(),
        }
    }
}
#[derive(Clone, Debug)]
pub enum ClassificationMetric {
    Accuracy,
    // BalancedAccuracy,
    // ConfusionMatrix,
    // HammingLoss,
    // JaccardScore,
    MatthewsCorrelationCoefficient,
    Precision,
    // RecallMatrix,
    RecallScore,
    // RocAucCurve, // Receiver Operating Characteristic Curve
    // RocAucScore, //Compute Area Under the Receiver Operating Characteristic Curve
    // SpecificityMatrix,
    // SpecificityScore,
    // ZeroOneLoss, //log loss?
    F1Score,
}
impl ClassificationMetric {
    fn calculate_metric(
        &self,
        prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
        test_folds: &[Dataset<usize, usize, Ix1>],
    ) -> Vec<f32> {
        match self {
            ClassificationMetric::Accuracy => calculate_accuracy(prediction_data, test_folds),
            ClassificationMetric::MatthewsCorrelationCoefficient => {
                calculate_mmc(prediction_data, test_folds)
            }
            ClassificationMetric::Precision => calculate_precision(prediction_data, test_folds),
            ClassificationMetric::RecallScore => {
                calculate_recall_score(prediction_data, test_folds)
            }
            ClassificationMetric::F1Score => calculate_f1_score(prediction_data, test_folds),
        }
    }

    fn direction(&self) -> MetricDirection {
        match self {
            ClassificationMetric::Accuracy => MetricDirection::Maximize,
            ClassificationMetric::MatthewsCorrelationCoefficient => MetricDirection::Maximize,
            ClassificationMetric::Precision => MetricDirection::OneBest,
            ClassificationMetric::RecallScore => MetricDirection::OneBest,
            ClassificationMetric::F1Score => MetricDirection::Maximize,
        }
    }
}

#[derive(Clone, Debug)]
pub enum RegressionMetric {
    MaxErrorSingleTarget,
    // MaxErrorMultiTarget,
    MSESingletarget,
    // MSEMultiTarget,
    MAESingleTarget,
    // MAEMultiTarget,
    R2ScoreSingleTarget,
    // R2ScoreMultiTarget,
    ExplainedVarianceSingleTarget,
    // ExplainedVarianceMultiTarget,
}
impl RegressionMetric {
    fn calculate_metric(
        &self,
        prediction_data: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
        test_folds: &[Dataset<f64, f64, Ix1>],
    ) -> Vec<f32> {
        match self {
            RegressionMetric::MaxErrorSingleTarget => {
                calculate_max_error(prediction_data, test_folds)
            }
            RegressionMetric::MSESingletarget => calculate_mse(prediction_data, test_folds),
            RegressionMetric::MAESingleTarget => calculate_mae(prediction_data, test_folds),
            RegressionMetric::R2ScoreSingleTarget => {
                calculate_r2_score(prediction_data, test_folds)
            }
            RegressionMetric::ExplainedVarianceSingleTarget => {
                calculate_explained_variance(prediction_data, test_folds)
            }
        }
    }

    fn direction(&self) -> MetricDirection {
        match self {
            RegressionMetric::MaxErrorSingleTarget => MetricDirection::Minimize,
            RegressionMetric::MSESingletarget => MetricDirection::Minimize,
            RegressionMetric::MAESingleTarget => MetricDirection::Minimize,
            RegressionMetric::R2ScoreSingleTarget => MetricDirection::OneBest,
            RegressionMetric::ExplainedVarianceSingleTarget => MetricDirection::Maximize,
        }
    }
}

pub fn calculate_classification_scores(
    fitness_metric: &Metric,
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    match fitness_metric {
        Metric::ClassificationMetric(classification_metric) => {
            classification_metric.calculate_metric(prediction_data, test_folds)
        }
        Metric::RegressionMetric(_) => {
            panic!("Regression metric in classification scoring")
        }
    }
}

pub fn calculate_regression_scores(
    fitness_metric: &Metric,
    prediction_data: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    match fitness_metric {
        Metric::ClassificationMetric(_) => {
            panic!("Classification metric in regression scoring")
        }
        Metric::RegressionMetric(regression_metric) => {
            regression_metric.calculate_metric(prediction_data, test_folds)
        }
    }
}

// Classification Metrics
fn calculate_accuracy(
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    let mut accuracies = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        let cm = &prediction.confusion_matrix(&test_folds[i]).unwrap();
        accuracies.push(cm.accuracy());
    }
    accuracies
}

fn calculate_recall_score(
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    let mut recalls = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        let cm = &prediction.confusion_matrix(&test_folds[i]).unwrap();
        recalls.push(cm.recall());
    }
    recalls
}

fn calculate_f1_score(
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    let mut f1_scores = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        let cm = &prediction.confusion_matrix(&test_folds[i]).unwrap();
        let mut score = cm.f1_score();
        if score.is_nan() {
            score = 0.7;
        }
        f1_scores.push(score);
    }
    f1_scores
}

fn calculate_mmc(
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    let mut mmcs = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        let cm = &prediction.confusion_matrix(&test_folds[i]).unwrap();
        mmcs.push(cm.mcc());
    }
    mmcs
}

fn calculate_precision(
    prediction_data: &Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<usize, usize, Ix1>],
) -> Vec<f32> {
    let mut precisions = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        let cm = &prediction.confusion_matrix(&test_folds[i]).unwrap();
        precisions.push(cm.precision());
    }
    precisions
}

// Regression Metrics
fn calculate_r2_score(
    prediction_data: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    let mut r2_scores = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        //let f_prediction = prediction.mapv(|x| x as f64);
        r2_scores.push(test_folds[i].r2(&prediction).unwrap() as f32);
    }
    r2_scores
}

fn calculate_mae(
    prediction_data: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    let mut maes = Vec::new();
    for (i, (prediction, _)) in enumerate(prediction_data) {
        //let f_prediction = prediction.mapv(|x| x as f32);
        maes.push(test_folds[i].mean_absolute_error(&prediction).unwrap() as f32);
    }
    maes
}

fn calculate_max_error(
    predictions: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    let mut max_errors = Vec::new();
    for (i, (prediction, _)) in enumerate(predictions) {
        max_errors.push(test_folds[i].max_error(&prediction).unwrap() as f32);
    }
    max_errors
}

fn calculate_mse(
    predictions: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    let mut mses = Vec::new();
    for (i, (prediction, _)) in enumerate(predictions) {
        mses.push(test_folds[i].mean_squared_error(&prediction).unwrap() as f32);
    }
    mses
}

fn calculate_explained_variance(
    predictions: &Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
    test_folds: &[Dataset<f64, f64, Ix1>],
) -> Vec<f32> {
    let mut explained_variances = Vec::new();
    for (i, (prediction, _)) in enumerate(predictions) {
        explained_variances.push(test_folds[i].explained_variance(&prediction).unwrap() as f32);
    }
    explained_variances
}
