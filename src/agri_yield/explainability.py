def shap_summary(model, X, max_display=12):
    import shap
    explainer = shap.Explainer(model.predict, X)
    values = explainer(X)
    return explainer, values, max_display
