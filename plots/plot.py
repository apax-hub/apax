
import subprocess




content1 = r"""\documentclass{standalone}
\usepackage{amsmath}
\usepackage{physics}
\begin{document}
$
"""

content2 = r"""$
\end{document}
"""

equations = [
    # r"y = f(x, \boldsymbol{\theta})",
    # r"\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - f(x_i; \theta) \right)^2",
    # r"\hat{y} \neq y",

    # r"f(x, \boldsymbol{\theta})",
    # r"p(y|x,\boldsymbol{\theta})",
    # r"p(y|x,\mathcal{D}) = \int p(y|x,\boldsymbol{\theta}) p(\boldsymbol{\theta}|\mathcal{D}) \,\mathrm{d}\boldsymbol{\theta}",
    # r"p(\boldsymbol{\theta}|\mathcal{D}) = \frac{1}{Z} \exp\left(- \mathcal{L}(\mathcal{D};\boldsymbol{\theta}) \right)",
    # r"\mathcal{L} = \mathrm{MSE}",

    # r"\boldsymbol{\theta}_{\text{MAP}}",
    # r"p(\boldsymbol{\theta} | \mathcal{D}) \approx \mathcal{N}(\boldsymbol{\theta}_{\mathrm{MAP}}, H^{-1})",

    # r"p(y|x,\mathcal{D}) \approx \mathcal{N}( f(x, \boldsymbol{\theta}_{\mathrm{MAP}}), \mathrm{Var} \left[ f(x, \boldsymbol{\theta})\right])",
    # r"\mathrm{Var} \left[ f(x, \boldsymbol{\theta})\right] = \grad_{\boldsymbol{\theta}} f(x, \boldsymbol{\theta}_{\mathrm{MAP}})^T H^{-1} \grad_{\boldsymbol{\theta}} f(x, \boldsymbol{\theta}_{\mathrm{MAP}})",
                                                                                                    

    # r"p(\boldsymbol{\theta} \mid \mathcal{D}) \approx \frac{1}{K} \sum_{k=1}^K \delta(\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{MAP}}^k)",
    # r"p(y \mid x, \mathcal{D}) = \frac{1}{K} \sum_{k=1}^K \delta\left( y - f_k(x, \boldsymbol{\theta}) \right)",

    # # r"p(y \mid x, \mathcal{D}) \approx \mathcal{N}\left( \bar{\mu}(x), \sigma^2(x) \right)",

    # # S10
    # r"\bar{E}  = \frac{1}{N_{\text{ens}}} \sum_m^{N_{\text{ens}}} E^{(m)}",
    # r"\sigma_E  = \sqrt{ \frac{1}{N_{\text{ens}} - 1} \sum_m^{N_{\text{ens}}} (E^{(m)} - \bar{E})^2 }",
    # r"\mathrm{NLL} = \frac{1}{2} \left[ \frac{(\bar{y} - \hat{y})^2}{\sigma_y^2} + \log 2 \pi \sigma_y^2 \right]",


    # # S13

    # r"\mathrm{Var} \left[ F_{ik} \right] = \boldsymbol{g}_{F_{ik}}^T H^{-1} \boldsymbol{g}_{F_{ik}}",
    # r"\boldsymbol{g}_{F_{ik}} = \frac{\partial F_{ik}}{\partial \boldsymbol{\theta}}",
    # r"F_i = \frac{\partial \bar{E}}{\partial r_i} = \frac{1}{N_{\text{ens}}} \sum_m^{N_{\text{ens}}} F_i^{(m)}",


    # r"p(y|x,\mathcal{D}) \approx \mathcal{N} \left(  \mathrm{Mean}\left[ f(x, \boldsymbol{\theta}^{\mathrm{SE}}_{\mathrm{MAP}}   \right]) ,  \mathrm{Var}\left[  f(x, \boldsymbol{\theta}^{\mathrm{SE}}_{\mathrm{MAP}}; \boldsymbol{\theta}^{\mathrm{SE, LL}}) \right]     \right)",

    # r"\boldsymbol{\theta}^{\mathrm{LL}} \propto \mathcal{N}(\boldsymbol{\theta}^{\mathrm{LL}}_{\mathrm{MAP}}, H^{-1})",
    # r"\boldsymbol{\theta}^{\mathrm{LL}} \propto \mathcal{N}(\boldsymbol{\theta}^{\mathrm{LL}}_{\mathrm{MAP}}, \sigma^2 \mathbf{I})",

    # r"\mathrm{NLL} = \frac{1}{2} \left[ \frac{(\bar{y} - y_{\text{ref}})^2}{\sigma_y^2} + \log 2 \pi \sigma_y^2 \right]",
    # r"\mathbf{f} = \nabla_{\theta_{\text{LL}}} E(\mathcal{S}, \theta), \ \mathbf{F} = [\mathbf{f}_1, \dots, \mathbf{f}_n]^\top"
    r"\sigma_y  = \frac{1}{N_{\text{ens}} - 1} \sum_k^{N_{\text{ens}}} (y^{(k)} - \bar{y})^2",


    



]

for ii, eq in enumerate(equations):
    string = content1 + eq + content2

    with open(f"{ii}.tex", "w") as f:
        f.write(string)
    subprocess.run(["pdflatex", f"{ii}.tex"]) 
    subprocess.run(["pdf2svg", f"{ii}.pdf", f"{ii}.svg"]) 
