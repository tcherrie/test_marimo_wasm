# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "scikit-fem",
#     "plotly",
# ]
# ///
# FICHIER MAÎTRE : contient à la fois la solution et la version étudiant.
# Les blocs "# >>> SOLUTION ... # <<< SOLUTION" sont retirés de la version étudiant,
# les blocs "# >>> ETUDIANT ... # <<< ETUDIANT" (commentés) y sont décommentés.
# Ne pas distribuer ce fichier : lancer `python tools/build_marimo.py` (depuis la racine du dépôt).

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Équation de la chaleur par éléments finis avec scikit-fem

    Dans ce notebook, on résout l'équation de la chaleur sur une plaque avec la **méthode des éléments finis**,
    à l'aide de la bibliothèque [scikit-fem](https://scikit-fem.readthedocs.io). Nous allons :

    1. résoudre le problème **stationnaire** $-\Delta u = f$ sur un carré, et vérifier la convergence de la méthode
       en comparant à une solution exacte (éléments P1 et P2) ;
    2. résoudre le problème **instationnaire** $\partial_t u - \alpha \Delta u = 0$ avec un θ-schéma
       (Euler explicite, Crank-Nicolson, Euler implicite) et étudier sa **stabilité** ;
    3. simuler une plaque chauffée localement et refroidie par **convection** sur ses bords
       (condition de Robin), et vérifier le **bilan d'énergie**.

    /// admonition | Comment travailler dans ce notebook
    * Ce notebook tourne **entièrement dans votre navigateur** : rien à installer.
      Le premier chargement de scikit-fem peut prendre quelques secondes.
    * Il est organisé en **trois colonnes** (faites défiler horizontalement) : problème stationnaire, problème instationnaire,
      application à une plaque refroidie.
    * Les cellules marquées **TODO** sont à compléter. Cliquez sur « 💡 Indice » pour obtenir de l'aide.
    * Le notebook est *réactif* : quand vous modifiez une cellule et l'exécutez (`Ctrl/Cmd + Entrée`),
      toutes les cellules qui en dépendent sont ré-exécutées automatiquement.
    * Votre code est conservé dans le navigateur, mais pensez à **télécharger votre notebook**
      (enregistrez avec `Ctrl/Cmd + S`, puis menu ☰ en haut à droite → *Export…* → *Python* → *Export notebook source*) pour le rendre.
    ///
    """)
    return


@app.cell
async def _():
    # Dans le navigateur (Pyodide), on s'assure que scikit-fem est chargé avant de l'importer
    import sys

    if sys.platform == "emscripten":
        import micropip

        await micropip.install("scikit-fem")

    from scipy.sparse.linalg import eigsh, splu
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriP1,
        ElementTriP2,
        FacetBasis,
        Functional,
        LinearForm,
        MeshTri,
        condense,
        solve,
    )
    from skfem.helpers import dot, grad

    return (
        Basis,
        BilinearForm,
        ElementTriP1,
        ElementTriP2,
        FacetBasis,
        Functional,
        LinearForm,
        MeshTri,
        condense,
        dot,
        eigsh,
        grad,
        solve,
        splu,
    )


@app.cell(hide_code=True)
def _(go, np):
    # Fonctions de tracé (fournies)
    def figure_maillage(mesh, titre="Maillage"):
        """Arêtes du maillage."""
        _f, _p = mesh.facets, mesh.p
        _nan = np.full(_f.shape[1], np.nan)
        _xs = np.column_stack([_p[0, _f[0]], _p[0, _f[1]], _nan]).ravel()
        _ys = np.column_stack([_p[1, _f[0]], _p[1, _f[1]], _nan]).ravel()
        _fig = go.Figure(go.Scatter(x=_xs, y=_ys, mode="lines", line=dict(color="rgb(23,15,48)", width=1)))
        _fig.update_layout(
            title=titre, width=420, height=420, plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(scaleanchor="x"), margin=dict(l=40, r=20, t=50, b=40), showlegend=False,
        )
        return _fig

    def figure_champ(mesh, u, titre, zmin=None, zmax=None, unite="u", largeur=480, hauteur=420, palette="Inferno"):
        """Champ nodal (valeurs aux sommets) sur un maillage structuré (MeshTri.init_tensor)."""
        _nv = mesh.p.shape[1]
        _fig = go.Figure(
            go.Contour(
                x=mesh.p[0], y=mesh.p[1], z=np.asarray(u)[:_nv],
                colorscale=palette, zmin=zmin, zmax=zmax, ncontours=25,
                contours=dict(coloring="heatmap"), line=dict(width=0.5),
                colorbar=dict(title=unite),
            )
        )
        _fig.update_layout(
            title=titre, width=largeur, height=hauteur, plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(scaleanchor="x"), margin=dict(l=40, r=20, t=50, b=40),
        )
        return _fig

    return figure_champ, figure_maillage


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problème stationnaire : formulation faible et assemblage

    On cherche la température $u$ sur le carré $\Omega = ]0, 1[^2$ telle que
    $$
    -\Delta u = f \quad \text{dans } \Omega, \qquad u = 0 \quad \text{sur } \partial\Omega .
    $$
    On choisit $f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$, de sorte que la solution exacte est
    $u_{\text{ex}}(x, y) = \sin(\pi x)\sin(\pi y)$ (solution *manufacturée*, pratique pour mesurer l'erreur).

    En multipliant par une fonction test $v$ nulle sur le bord et en intégrant par parties, on obtient la
    **formulation faible** : trouver $u$ telle que, pour tout $v$,
    $$
    \underbrace{\int_\Omega \nabla u \cdot \nabla v \, \mathrm{d}x}_{a(u, v)}
    = \underbrace{\int_\Omega f\, v \, \mathrm{d}x}_{\ell(v)} .
    $$
    En décomposant $u_h = \sum_j U_j \varphi_j$ sur les fonctions de forme $\varphi_j$ du maillage, on obtient
    le système linéaire $K\,U = F$, avec $K_{ij} = a(\varphi_j, \varphi_i)$ (matrice de **rigidité**) et
    $F_i = \ell(\varphi_i)$.

    Dans scikit-fem, une forme s'écrit comme une fonction Python décorée par `@BilinearForm` (arguments `u, v, w`)
    ou `@LinearForm` (arguments `v, w`). Elle est évaluée aux points de quadrature ; `w.x` contient leurs coordonnées.

    ### TODO 1 : formes bilinéaire et linéaire

    Complétez `laplace` (la forme $a$) et `source_poisson` (la forme $\ell$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Indice 1": mo.md("`from skfem.helpers import dot, grad` est déjà importé : `dot(grad(u), grad(v))` est le produit scalaire des gradients."),
            "💡 Indice 2": mo.md("Dans `source_poisson`, `x, y = w.x` donne les coordonnées des points de quadrature. Renvoyez `f(x, y) * v` avec `np.sin` et `np.pi`."),
        }
    )
    return


@app.cell
def _(BilinearForm, LinearForm, dot, grad, np):
    @BilinearForm
    def laplace(u, v, w):
        # >>> ETUDIANT
        # # TODO 1
        # return None
        # <<< ETUDIANT
        # >>> SOLUTION
        return dot(grad(u), grad(v))
        # <<< SOLUTION

    @LinearForm
    def source_poisson(v, w):
        x, y = w.x
        # >>> ETUDIANT
        # # TODO 1
        # return None
        # <<< ETUDIANT
        # >>> SOLUTION
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * v
        # <<< SOLUTION

    return laplace, source_poisson


@app.cell(hide_code=True)
def _(Basis, ElementTriP1, MeshTri, laplace, mo, np, source_poisson):
    # Vérification automatique du TODO 1
    def _verifier():
        _m = MeshTri.init_tensor(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
        _b = Basis(_m, ElementTriP1())
        try:
            _K = laplace.assemble(_b)
            _F = source_poisson.assemble(_b)
        except Exception as _e:
            if "None" in repr(_e) or isinstance(_e, TypeError):
                return False, mo.callout(mo.md("**TODO 1** pas encore complété."), kind="warn")
            return False, mo.callout(mo.md(f"❌ L'assemblage a levé une erreur : `{_e!r}`"), kind="danger")
        _msgs = []
        if abs(_K - _K.T).max() > 1e-10:
            _msgs.append("`K` n'est pas symétrique")
        if np.abs(_K @ np.ones(_K.shape[0])).max() > 1e-8 or _K.diagonal().min() <= 0:
            _msgs.append("`K @ 1` devrait être nul (le gradient d'une constante est nul) et la diagonale positive")
        # ∫ f = 2π² (2/π)² = 8
        if abs(_F.sum() - 8.0) > 0.4:
            _msgs.append(f"la somme des composantes de `F` vaut `{_F.sum():.3f}` au lieu de ≈ 8 (= ∫f)")
        if _msgs:
            return False, mo.callout(mo.md("❌ " + " ; ".join(_msgs) + "."), kind="danger")
        return True, mo.callout(mo.md("✅ `laplace` et `source_poisson` sont corrects."), kind="success")

    ok1, _msg = _verifier()
    _msg
    return (ok1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TODO 2 : assembler et résoudre

    Complétez `resoudre_poisson(n, element)`, qui résout le problème sur un maillage de $n \times n$ carrés
    (chacun coupé en deux triangles) :

    1. assemblez `K` et `F` ;
    2. récupérez les degrés de liberté du bord `D` (où l'on impose $u = 0$) ;
    3. éliminez-les du système et résolvez pour obtenir le vecteur `u` complet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Indice 1": mo.md("`K = laplace.assemble(basis)` et `F = source_poisson.assemble(basis)`."),
            "💡 Indice 2": mo.md("Sans argument, `basis.get_dofs()` renvoie les degrés de liberté situés sur tout le bord du domaine."),
            "💡 Indice 3": mo.md("`condense(K, F, D=D)` élimine les ddl `D` (valeur imposée nulle par défaut) et `solve(*condense(...))` renvoie le vecteur solution complet."),
        }
    )
    return


@app.cell
def _(Basis, MeshTri, condense, laplace, np, solve, source_poisson):
    def resoudre_poisson(n, element):
        """Résout -Δu = f, u = 0 au bord, sur [0,1]² maillé en n x n carrés. Renvoie (basis, u)."""
        _x = np.linspace(0, 1, n + 1)
        mesh = MeshTri.init_tensor(_x, _x)
        basis = Basis(mesh, element)
        # >>> ETUDIANT
        # # TODO 2
        # K = None
        # F = None
        # D = None
        # u = None
        # <<< ETUDIANT
        # >>> SOLUTION
        K = laplace.assemble(basis)
        F = source_poisson.assemble(basis)
        D = basis.get_dofs()
        u = solve(*condense(K, F, D=D))
        # <<< SOLUTION
        return basis, u

    return (resoudre_poisson,)


@app.cell(hide_code=True)
def _(Basis, Functional, grad, np):
    # Erreurs L² et H¹ (semi-norme) par rapport à la solution exacte (fourni)
    @Functional
    def _err_L2(w):
        x, y = w.x
        return (w["uh"] - np.sin(np.pi * x) * np.sin(np.pi * y)) ** 2

    @Functional
    def _err_H1(w):
        x, y = w.x
        _gx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        _gy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        _g = grad(w["uh"])
        return (_g[0] - _gx) ** 2 + (_g[1] - _gy) ** 2

    def erreurs(basis, u):
        """Renvoie (erreur L², erreur H¹) en intégrant avec une quadrature précise."""
        _b = Basis(basis.mesh, basis.elem, intorder=8)
        _uh = _b.interpolate(u)
        return np.sqrt(_err_L2.assemble(_b, uh=_uh)), np.sqrt(_err_H1.assemble(_b, uh=_uh))

    return (erreurs,)


@app.cell(hide_code=True)
def _(ElementTriP1, erreurs, mo, ok1, resoudre_poisson):
    # Vérification automatique du TODO 2
    def _verifier():
        if not ok1:
            return False, mo.callout(mo.md("Terminez d'abord le **TODO 1**."), kind="warn")
        try:
            _b, _u = resoudre_poisson(16, ElementTriP1())
        except Exception as _e:
            return False, mo.callout(mo.md(f"❌ `resoudre_poisson` a levé une erreur : `{_e!r}`"), kind="danger")
        if _u is None:
            return False, mo.callout(mo.md("**TODO 2** pas encore complété."), kind="warn")
        if len(_u) != _b.N:
            return False, mo.callout(mo.md(f"❌ `u` a {len(_u)} composantes au lieu de {_b.N} : renvoyez le vecteur **complet** (avec les ddl du bord)."), kind="danger")
        _eL2, _ = erreurs(_b, _u)
        if _eL2 < 1e-2:
            return True, mo.callout(mo.md(f"✅ `resoudre_poisson` est correct (erreur L² = `{_eL2:.2e}` pour n = 16)."), kind="success")
        return False, mo.callout(mo.md(f"❌ Erreur L² trop grande (`{_eL2:.2e}`). Avez-vous bien imposé u = 0 sur le bord ?"), kind="danger")

    ok2, _msg = _verifier()
    _msg
    return (ok2,)


@app.cell
def _(mo):
    n_poisson = mo.ui.slider(2, 48, value=8, label="Nombre de mailles par côté n", show_value=True)
    elem_poisson = mo.ui.dropdown(["P1", "P2"], value="P1", label="Élément")
    mo.hstack([n_poisson, elem_poisson], justify="start", gap=2)
    return elem_poisson, n_poisson


@app.cell(hide_code=True)
def _(
    ElementTriP1,
    ElementTriP2,
    elem_poisson,
    erreurs,
    figure_champ,
    figure_maillage,
    mo,
    n_poisson,
    np,
    ok2,
    resoudre_poisson,
):
    mo.stop(not ok2, mo.callout(mo.md("Complétez les **TODO 1 et 2** pour afficher la solution."), kind="warn"))
    _elem = {"P1": ElementTriP1(), "P2": ElementTriP2()}[elem_poisson.value]
    _basis, _u = resoudre_poisson(n_poisson.value, _elem)
    _mesh = _basis.mesh
    _x, _y = _mesh.p
    _ecart = _u[: _mesh.p.shape[1]] - np.sin(np.pi * _x) * np.sin(np.pi * _y)
    _eL2, _eH1 = erreurs(_basis, _u)
    mo.vstack(
        [
            mo.md(
                f"**{_basis.N} degrés de liberté**, {_mesh.t.shape[1]} triangles · "
                f"erreur L² = `{_eL2:.3e}` · erreur H¹ = `{_eH1:.3e}`"
            ),
            mo.hstack(
                [
                    figure_maillage(_mesh),
                    figure_champ(_mesh, _u, "Solution éléments finis u_h", unite="u"),
                    figure_champ(_mesh, _ecart, "Écart aux sommets u_h − u_ex", unite="écart", palette="RdBu"),
                ],
                wrap=True,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence en maillage

    La théorie (lemme de Céa, argument de Aubin-Nitsche) prévoit, pour des éléments de degré $p$ et une solution régulière :
    $$
    \|u - u_h\|_{L^2} \le C\, h^{p+1}, \qquad |u - u_h|_{H^1} \le C'\, h^{p},
    $$
    où $h$ est la taille des mailles. En échelle log-log, l'erreur est donc une droite de pente $p+1$ (resp. $p$).

    ### TODO 3 : ordre de convergence

    Complétez `ordre_convergence(h, err)` qui renvoie la **pente** de la droite des moindres carrés
    passant par les points $(\log h_i, \log e_i)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Indice": mo.md("`np.polyfit(X, Y, 1)` renvoie `[pente, ordonnée à l'origine]` de la droite ajustée sur les points (X, Y)."),
        }
    )
    return


@app.cell
def _(np):
    def ordre_convergence(h, err):
        """Pente de log(err) en fonction de log(h)."""
        h = np.asarray(h, dtype=float)
        err = np.asarray(err, dtype=float)
        # >>> ETUDIANT
        # # TODO 3
        # pente = None
        # <<< ETUDIANT
        # >>> SOLUTION
        pente = np.polyfit(np.log(h), np.log(err), 1)[0]
        # <<< SOLUTION
        return pente

    return (ordre_convergence,)


@app.cell(hide_code=True)
def _(mo, np, ordre_convergence):
    # Vérification automatique du TODO 3
    def _verifier():
        _h = np.array([0.2, 0.1, 0.05, 0.025])
        try:
            _p = ordre_convergence(_h, 3 * _h**2.5)
        except Exception as _e:
            return False, mo.callout(mo.md(f"❌ `ordre_convergence` a levé une erreur : `{_e!r}`"), kind="danger")
        if _p is None:
            return False, mo.callout(mo.md("**TODO 3** pas encore complété."), kind="warn")
        if np.isclose(_p, 2.5):
            return True, mo.callout(mo.md("✅ `ordre_convergence` est correct."), kind="success")
        return False, mo.callout(mo.md(f"❌ Pour err = 3 h^2,5, on attend une pente de 2,5 ; obtenu `{_p}`."), kind="danger")

    ok3, _msg = _verifier()
    _msg
    return (ok3,)


@app.cell(hide_code=True)
def _(ElementTriP1, ElementTriP2, erreurs, go, mo, ok2, ok3, ordre_convergence, resoudre_poisson):
    mo.stop(not (ok2 and ok3), mo.callout(mo.md("Complétez les **TODO 2 et 3** pour lancer l'étude de convergence."), kind="warn"))
    _ns = [4, 8, 16, 32]
    _h = [1 / _n for _n in _ns]
    _fig = go.Figure()
    _lignes = []
    _couleurs = {"P1": "rgb(58,70,245)", "P2": "rgb(245,70,58)"}
    for _nom, _elem in (("P1", ElementTriP1()), ("P2", ElementTriP2())):
        _eL2, _eH1 = [], []
        for _n in _ns:
            _b, _u = resoudre_poisson(_n, _elem)
            _a, _c = erreurs(_b, _u)
            _eL2.append(_a)
            _eH1.append(_c)
        _fig.add_trace(go.Scatter(x=_h, y=_eL2, mode="lines+markers", name=f"{_nom}, erreur L²", line=dict(color=_couleurs[_nom])))
        _fig.add_trace(go.Scatter(x=_h, y=_eH1, mode="lines+markers", name=f"{_nom}, erreur H¹", line=dict(color=_couleurs[_nom], dash="dash")))
        _lignes.append(f"| {_nom} | {ordre_convergence(_h, _eL2):.2f} | {ordre_convergence(_h, _eH1):.2f} |")
    _fig.update_layout(
        title="Convergence en maillage", xaxis=dict(type="log", title="h"), yaxis=dict(type="log", title="erreur"),
        plot_bgcolor="rgba(0,0,0,0)", width=650, height=420,
    )
    mo.vstack(
        [
            _fig,
            mo.md("| Élément | ordre L² mesuré | ordre H¹ mesuré |\n|---|---|---|\n" + "\n".join(_lignes)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Questions
    1. Les ordres mesurés sont-ils conformes à la théorie pour P1 et P2 ?
    2. À nombre de degrés de liberté égal, quel élément est le plus précis ? (Regardez le nombre de ddl affiché au-dessus
       pour P1 avec n = 32 et P2 avec n = 16.)
    3. Avec le curseur, observez l'écart $u_h - u_{\text{ex}}$ aux sommets en P1. Où l'erreur est-elle la plus grande ?
    ///
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Problème instationnaire : θ-schéma et stabilité

    On considère maintenant l'équation de la chaleur
    $$
    \partial_t u - \alpha \Delta u = 0 \ \text{ dans } \Omega, \qquad u = 0 \ \text{ sur } \partial\Omega,
    \qquad u(x, y, 0) = \sin(\pi x)\sin(\pi y),
    $$
    avec la diffusivité $\alpha = 1$. La solution exacte décroît exponentiellement :
    $u(x, y, t) = e^{-2\pi^2 \alpha t}\sin(\pi x)\sin(\pi y)$.

    **Semi-discrétisation en espace.** La formulation faible donne le système d'équations différentielles
    $$
    M\, \dot U + \alpha K\, U = 0, \qquad M_{ij} = \int_\Omega \varphi_j \varphi_i \,\mathrm{d}x \ \ \text{(matrice de masse)} .
    $$

    **Discrétisation en temps (θ-schéma).** Avec un pas $\Delta t$ et $\theta \in [0, 1]$ :
    $$
    \underbrace{(M + \theta\,\Delta t\,\alpha K)}_{A}\, U^{n+1} = \underbrace{(M - (1-\theta)\,\Delta t\,\alpha K)}_{B}\, U^n .
    $$
    $\theta = 0$ : Euler explicite ; $\theta = 1/2$ : Crank-Nicolson ; $\theta = 1$ : Euler implicite.

    ### TODO 4 : matrice de masse

    Complétez la forme bilinéaire `masse`.
    """)
    return


@app.cell
def _(BilinearForm):
    @BilinearForm
    def masse(u, v, w):
        # >>> ETUDIANT
        # # TODO 4
        # return None
        # <<< ETUDIANT
        # >>> SOLUTION
        return u * v
        # <<< SOLUTION

    return (masse,)


@app.cell(hide_code=True)
def _(Basis, ElementTriP1, MeshTri, masse, mo, np):
    # Vérification automatique du TODO 4
    def _verifier():
        _m = MeshTri.init_tensor(np.linspace(0, 2, 7), np.linspace(0, 1, 4))
        _b = Basis(_m, ElementTriP1())
        try:
            _M = masse.assemble(_b)
        except Exception as _e:
            if "None" in repr(_e) or isinstance(_e, TypeError):
                return False, mo.callout(mo.md("**TODO 4** pas encore complété."), kind="warn")
            return False, mo.callout(mo.md(f"❌ L'assemblage a levé une erreur : `{_e!r}`"), kind="danger")
        # Somme des coefficients = ∫ 1 · 1 = aire du domaine (2)
        if np.isclose(_M.sum(), 2.0) and _M.min() >= 0:
            return True, mo.callout(mo.md("✅ `masse` est correcte."), kind="success")
        return False, mo.callout(mo.md(f"❌ La somme des coefficients de `M` doit valoir l'aire du domaine (2), obtenu `{_M.sum():.4f}`."), kind="danger")

    ok4, _msg = _verifier()
    _msg
    return (ok4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TODO 5 : matrices du θ-schéma

    Complétez `matrices_theta(M, K, dt, theta, alpha)` qui renvoie le couple `(A, B)` défini ci-dessus.
    Les matrices sont des matrices creuses SciPy : on les combine avec `+`, `-` et `*` comme des tableaux.

    La boucle en temps (fournie, fonction `integrer`) factorise $A$ une seule fois (factorisation LU creuse),
    puis résout $A U^{n+1} = B U^n$ à chaque pas.
    """)
    return


@app.cell
def _():
    def matrices_theta(M, K, dt, theta, alpha):
        """Matrices (A, B) du θ-schéma : A U^{n+1} = B U^n."""
        # >>> ETUDIANT
        # # TODO 5
        # A = None
        # B = None
        # <<< ETUDIANT
        # >>> SOLUTION
        A = M + theta * dt * alpha * K
        B = M - (1 - theta) * dt * alpha * K
        # <<< SOLUTION
        return A, B

    return (matrices_theta,)


@app.cell(hide_code=True)
def _(matrices_theta, mo, np):
    # Vérification automatique du TODO 5
    import scipy.sparse as _sp

    def _verifier():
        _M = _sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        _K = _sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        try:
            _A, _B = matrices_theta(_M, _K, 0.1, 0.25, 2.0)
        except Exception as _e:
            return False, mo.callout(mo.md(f"❌ `matrices_theta` a levé une erreur : `{_e!r}`"), kind="danger")
        if _A is None or _B is None:
            return False, mo.callout(mo.md("**TODO 5** pas encore complété."), kind="warn")
        _A = _A.toarray() if hasattr(_A, "toarray") else np.asarray(_A)
        _B = _B.toarray() if hasattr(_B, "toarray") else np.asarray(_B)
        if np.allclose(_A, [[2.05, 0.95], [0.95, 2.05]]) and np.allclose(_B, [[1.85, 1.15], [1.15, 1.85]]):
            return True, mo.callout(mo.md("✅ `matrices_theta` est correct."), kind="success")
        return False, mo.callout(mo.md("❌ Résultat incorrect. Vérifiez les facteurs θ, (1 − θ), Δt et α."), kind="danger")

    ok5, _msg = _verifier()
    _msg
    return (ok5,)


@app.cell(hide_code=True)
def _(np, splu):
    def integrer(A, B, u0, n_pas, G=None, I=None, seuil=1e6):
        """Intègre A U^{n+1} = B U^n + G sur n_pas pas de temps (fourni).

        I : indices des ddl inconnus (les autres restent nuls, condition de Dirichlet homogène).
        Renvoie (U, stable) avec U de forme (nombre de pas calculés + 1, N).
        L'intégration s'arrête si la solution dépasse `seuil` (instabilité numérique).
        """
        _N = len(u0)
        I = np.arange(_N) if I is None else I
        _lu = splu(A[I][:, I].tocsc())
        _B = B[I][:, I]
        _G = np.zeros(len(I)) if G is None else np.asarray(G)[I]
        U = np.zeros((n_pas + 1, _N))
        U[0] = u0
        _u = np.asarray(u0, dtype=float)[I]
        for _n in range(n_pas):
            _u = _lu.solve(_B @ _u + _G)
            U[_n + 1, I] = _u
            if not np.all(np.isfinite(_u)) or np.abs(_u).max() > seuil:
                return U[: _n + 2], False
        return U, True

    return (integrer,)


@app.cell
def _(mo):
    theta = mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="θ", show_value=True)
    dt_choix = mo.ui.dropdown(
        ["0.0001", "0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01", "0.02"],
        value="0.002", label="Pas de temps Δt",
    )
    n_chaleur = mo.ui.slider(4, 32, step=4, value=16, label="Mailles par côté n", show_value=True)
    t_final = mo.ui.slider(0.02, 0.3, step=0.02, value=0.1, label="Durée simulée", show_value=True)
    mo.vstack([mo.md("### Paramètres de la simulation"), theta, dt_choix, n_chaleur, t_final])
    return dt_choix, n_chaleur, t_final, theta


@app.cell
def _(
    Basis,
    ElementTriP1,
    MeshTri,
    dt_choix,
    integrer,
    laplace,
    masse,
    matrices_theta,
    mo,
    n_chaleur,
    np,
    ok1,
    ok4,
    ok5,
    t_final,
    theta,
):
    mo.stop(not (ok1 and ok4 and ok5), mo.callout(mo.md("Complétez les **TODO 1, 4 et 5** pour lancer la simulation."), kind="warn"))
    alpha = 1.0
    dt = float(dt_choix.value)
    n_pas = int(round(t_final.value / dt))

    _x = np.linspace(0, 1, n_chaleur.value + 1)
    mesh_ch = MeshTri.init_tensor(_x, _x)
    basis_ch = Basis(mesh_ch, ElementTriP1())
    K_ch = laplace.assemble(basis_ch)
    M_ch = masse.assemble(basis_ch)
    I_ch = basis_ch.complement_dofs(basis_ch.get_dofs())  # ddl intérieurs

    _xd, _yd = basis_ch.doflocs
    u0 = np.sin(np.pi * _xd) * np.sin(np.pi * _yd)

    _A, _B = matrices_theta(M_ch, K_ch, dt, theta.value, alpha)
    U_ch, stable = integrer(_A, _B, u0, n_pas, I=I_ch)
    temps_ch = dt * np.arange(len(U_ch))
    return I_ch, K_ch, M_ch, U_ch, alpha, dt, mesh_ch, n_pas, stable, temps_ch


@app.cell
def _(U_ch, mo):
    instant = mo.ui.slider(0, len(U_ch) - 1, value=len(U_ch) - 1, label="Pas de temps affiché", show_value=True)
    instant
    return (instant,)


@app.cell(hide_code=True)
def _(U_ch, alpha, figure_champ, go, instant, mesh_ch, mo, n_pas, np, stable, temps_ch):
    _k = instant.value
    _champ = figure_champ(
        mesh_ch, U_ch[_k], f"Température à t = {temps_ch[_k]:.4f}",
        zmin=0 if stable else None, zmax=1 if stable else None,
    )
    _amp = np.abs(U_ch).max(axis=1)
    _t_fin = np.linspace(0, temps_ch[-1], 200)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_t_fin, y=np.exp(-2 * np.pi**2 * alpha * _t_fin), mode="lines", name="exacte", line=dict(color="black", dash="dash")))
    _fig.add_trace(go.Scatter(x=temps_ch, y=_amp, mode="lines+markers", marker=dict(size=3), name="éléments finis", line=dict(color="rgb(245,70,58)")))
    _fig.add_vline(x=temps_ch[_k], line=dict(color="gray", width=1))
    _fig.update_layout(
        title="Amplitude max |u| au cours du temps", xaxis_title="t", yaxis=dict(type="log", title="max |u|"),
        plot_bgcolor="rgba(0,0,0,0)", width=480, height=420,
    )
    _alerte = (
        mo.callout(mo.md(f"⚠️ **Instabilité numérique** : la solution a explosé après {len(U_ch) - 1} pas sur {n_pas}. Réduisez Δt ou augmentez θ."), kind="danger")
        if not stable
        else mo.md(f"Erreur relative sur l'amplitude finale : `{abs(_amp[-1] / np.exp(-2 * np.pi**2 * alpha * temps_ch[-1]) - 1):.2e}`")
    )
    mo.vstack([_alerte, mo.hstack([_champ, _fig], wrap=True)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TODO 6 : pas de temps critique

    Pour un mode propre $M \Phi = \lambda^{-1} K \Phi$ (c'est-à-dire $K\Phi = \lambda M\Phi$), le θ-schéma multiplie
    l'amplitude du mode à chaque pas par le **facteur d'amplification**
    $$
    g(\lambda) = \frac{1 - (1-\theta)\,\alpha\lambda\,\Delta t}{1 + \theta\,\alpha\lambda\,\Delta t} .
    $$
    Le schéma est stable si $|g| \le 1$ pour toutes les valeurs propres, la plus contraignante étant $\lambda_{\max}$.
    On montre que :

    * si $\theta \ge 1/2$, le schéma est **inconditionnellement stable** ;
    * si $\theta < 1/2$, il faut $\Delta t \le \Delta t_c = \dfrac{2}{(1 - 2\theta)\,\alpha\,\lambda_{\max}}$.

    Complétez `dt_critique(lam_max, theta, alpha)`, qui renvoie $\Delta t_c$ (et `np.inf` si $\theta \ge 1/2$).
    La valeur propre $\lambda_{\max}$ est calculée pour vous avec `scipy.sparse.linalg.eigsh`.
    """)
    return


@app.cell
def _(np):
    def dt_critique(lam_max, theta, alpha):
        """Pas de temps critique du θ-schéma (np.inf si inconditionnellement stable)."""
        # >>> ETUDIANT
        # # TODO 6
        # dtc = None
        # <<< ETUDIANT
        # >>> SOLUTION
        if theta >= 0.5:
            dtc = np.inf
        else:
            dtc = 2.0 / ((1 - 2 * theta) * alpha * lam_max)
        # <<< SOLUTION
        return dtc

    return (dt_critique,)


@app.cell(hide_code=True)
def _(dt_critique, mo, np):
    # Vérification automatique du TODO 6
    def _verifier():
        try:
            _a = dt_critique(1000.0, 0.0, 2.0)
            _b = dt_critique(1000.0, 0.25, 1.0)
            _c = dt_critique(1000.0, 0.5, 1.0)
        except Exception as _e:
            return False, mo.callout(mo.md(f"❌ `dt_critique` a levé une erreur : `{_e!r}`"), kind="danger")
        if _a is None:
            return False, mo.callout(mo.md("**TODO 6** pas encore complété."), kind="warn")
        if np.isclose(_a, 1e-3) and np.isclose(_b, 4e-3) and np.isinf(_c):
            return True, mo.callout(mo.md("✅ `dt_critique` est correct."), kind="success")
        return False, mo.callout(mo.md("❌ Résultat incorrect. Pensez au cas θ ≥ 1/2."), kind="danger")

    ok6, _msg = _verifier()
    _msg
    return (ok6,)


@app.cell(hide_code=True)
def _(I_ch, K_ch, M_ch, alpha, dt, dt_critique, eigsh, mo, ok6, theta):
    mo.stop(not ok6, mo.callout(mo.md("Complétez le **TODO 6** pour comparer Δt au pas critique."), kind="warn"))
    _K = K_ch[I_ch][:, I_ch]
    _M = M_ch[I_ch][:, I_ch]
    lam_max = float(eigsh(_K, k=1, M=_M, which="LM", return_eigenvectors=False, tol=1e-4)[0])
    _dtc = dt_critique(lam_max, theta.value, alpha)
    _txt = "∞ (inconditionnellement stable)" if _dtc == float("inf") else f"{_dtc:.2e}"
    mo.callout(
        mo.md(
            f"λ_max = `{lam_max:.4g}` · pas critique Δt_c = **{_txt}** · pas choisi Δt = `{dt:g}` → "
            + ("**stable**" if dt <= _dtc else "**instable** attendu")
        ),
        kind="success" if dt <= _dtc else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Questions
    1. Avec θ = 0 (Euler explicite), cherchez le plus grand Δt stable. Est-il cohérent avec $\Delta t_c$ ?
       Que devient $\Delta t_c$ quand on divise la taille de maille par 2 ? Montrez que $\lambda_{\max} \propto h^{-2}$.
    2. Avec θ = 0,5 (Crank-Nicolson) et un grand pas (Δt = 0,02), le schéma est stable mais la solution présente-t-elle
       un comportement physique ? Calculez $g(\lambda_{\max})$ dans ce cas et expliquez.
    3. Comparez l'erreur sur l'amplitude finale pour θ = 1 et θ = 0,5 en divisant Δt par 2.
       Quels ordres de convergence en temps retrouvez-vous ?
    ///
    """)
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Plaque chauffée et refroidie par convection

    Une plaque rectangulaire $\Omega = ]0, 2[ \times ]0, 1[$, initialement à la température ambiante ($u = 0$),
    est chauffée par une source localisée (une résistance) de puissance volumique
    $q(x, y) = Q\, \exp\!\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2}\right)$.
    Ses bords échangent de la chaleur avec l'air ambiant par **convection** (loi de Newton). En variables adimensionnées :
    $$
    \partial_t u - \Delta u = q \ \text{ dans } \Omega, \qquad
    -\nabla u \cdot \mathbf{n} = \mathrm{Bi}\; u \ \text{ sur } \partial\Omega \quad \text{(condition de Robin)},
    $$
    où le nombre de Biot $\mathrm{Bi} = hL/k$ compare la convection à la conduction.

    Cette fois, la température n'est pas imposée au bord : l'intégration par parties fait apparaître un **terme de bord**
    $$
    \int_\Omega \partial_t u\, v + \int_\Omega \nabla u \cdot \nabla v + \mathrm{Bi} \int_{\partial\Omega} u\, v = \int_\Omega q\, v ,
    $$
    soit $M\dot U + (K + \mathrm{Bi}\,R)\,U = F$, avec $R_{ij} = \int_{\partial\Omega} \varphi_j \varphi_i \,\mathrm{d}s$.

    Dans scikit-fem, une intégrale sur le bord s'assemble avec une `FacetBasis` au lieu d'une `Basis` :
    la forme bilinéaire s'écrit de la même façon, mais elle est évaluée aux points de quadrature des arêtes du bord.

    ### TODO 7 : terme de convection

    Complétez la forme bilinéaire `bord` (intégrande de $R$, sans le facteur Bi).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Indice": mo.md("C'est la même intégrande que la matrice de masse : seule la base (`FacetBasis`) change le domaine d'intégration."),
        }
    )
    return


@app.cell
def _(BilinearForm):
    @BilinearForm
    def bord(u, v, w):
        # >>> ETUDIANT
        # # TODO 7
        # return None
        # <<< ETUDIANT
        # >>> SOLUTION
        return u * v
        # <<< SOLUTION

    return (bord,)


@app.cell(hide_code=True)
def _(ElementTriP1, FacetBasis, MeshTri, bord, mo, np):
    # Vérification automatique du TODO 7
    def _verifier():
        _m = MeshTri.init_tensor(np.linspace(0, 2, 9), np.linspace(0, 1, 5))
        _fb = FacetBasis(_m, ElementTriP1())
        try:
            _R = bord.assemble(_fb)
        except Exception as _e:
            if "None" in repr(_e) or isinstance(_e, TypeError):
                return False, mo.callout(mo.md("**TODO 7** pas encore complété."), kind="warn")
            return False, mo.callout(mo.md(f"❌ L'assemblage a levé une erreur : `{_e!r}`"), kind="danger")
        # Somme des coefficients = ∫_∂Ω 1 = périmètre (6)
        if np.isclose(_R.sum(), 6.0) and _R.min() >= 0:
            return True, mo.callout(mo.md("✅ `bord` est correcte."), kind="success")
        return False, mo.callout(mo.md(f"❌ La somme des coefficients de `R` doit valoir le périmètre (6), obtenu `{_R.sum():.4f}`."), kind="danger")

    ok7, _msg = _verifier()
    _msg
    return (ok7,)


@app.cell
def _(mo):
    biot = mo.ui.slider(0.1, 20.0, step=0.1, value=2.0, label="Nombre de Biot Bi", show_value=True)
    x0 = mo.ui.slider(0.1, 1.9, step=0.05, value=0.5, label="Position x₀ de la source", show_value=True)
    y0 = mo.ui.slider(0.1, 0.9, step=0.05, value=0.5, label="Position y₀ de la source", show_value=True)
    puissance = mo.ui.slider(10.0, 200.0, step=10.0, value=100.0, label="Intensité Q", show_value=True)
    duree = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="Durée simulée", show_value=True)
    mo.vstack([mo.md("### Paramètres de la plaque"), biot, x0, y0, puissance, duree])
    return biot, duree, puissance, x0, y0


@app.cell
def _(
    Basis,
    ElementTriP1,
    FacetBasis,
    LinearForm,
    MeshTri,
    biot,
    bord,
    duree,
    integrer,
    laplace,
    masse,
    matrices_theta,
    mo,
    np,
    ok1,
    ok4,
    ok5,
    ok7,
    puissance,
    solve,
    x0,
    y0,
):
    mo.stop(not (ok1 and ok4 and ok5 and ok7), mo.callout(mo.md("Complétez les **TODO 1, 4, 5 et 7** pour simuler la plaque."), kind="warn"))
    sigma_src = 0.08
    xs_src, ys_src, Q_src, Bi_val = x0.value, y0.value, puissance.value, biot.value

    @LinearForm
    def source_plaque(v, w):
        x, y = w.x
        return Q_src * np.exp(-((x - xs_src) ** 2 + (y - ys_src) ** 2) / (2 * sigma_src**2)) * v

    mesh_pl = MeshTri.init_tensor(np.linspace(0, 2, 49), np.linspace(0, 1, 25))
    _basis = Basis(mesh_pl, ElementTriP1())
    _fbasis = FacetBasis(mesh_pl, ElementTriP1())
    M_pl = masse.assemble(_basis)
    R_pl = bord.assemble(_fbasis)
    Kt_pl = laplace.assemble(_basis) + Bi_val * R_pl
    F_pl = source_plaque.assemble(_basis)

    # Euler implicite (θ = 1), 150 pas
    n_pas_pl = 150
    dt_pl = duree.value / n_pas_pl
    _A, _B = matrices_theta(M_pl, Kt_pl, dt_pl, 1.0, 1.0)
    U_pl, _ = integrer(_A, _B, np.zeros(_basis.N), n_pas_pl, G=dt_pl * F_pl)
    temps_pl = dt_pl * np.arange(n_pas_pl + 1)

    # Régime permanent : (K + Bi R) U = F
    U_perm = solve(Kt_pl, F_pl)
    i_sonde = int(np.argmin((_basis.doflocs[0] - xs_src) ** 2 + (_basis.doflocs[1] - ys_src) ** 2))
    return F_pl, M_pl, R_pl, U_perm, U_pl, dt_pl, i_sonde, mesh_pl, temps_pl


@app.cell
def _(U_pl, mo):
    instant_pl = mo.ui.slider(0, len(U_pl) - 1, value=len(U_pl) - 1, label="Pas de temps affiché", show_value=True)
    instant_pl
    return (instant_pl,)


@app.cell(hide_code=True)
def _(U_perm, U_pl, figure_champ, go, i_sonde, instant_pl, mesh_pl, mo, temps_pl):
    _k = instant_pl.value
    _zmax = float(U_perm.max())
    _champ = figure_champ(mesh_pl, U_pl[_k], f"Température à t = {temps_pl[_k]:.3f}", zmin=0, zmax=_zmax, largeur=620, hauteur=360)
    _perm = figure_champ(mesh_pl, U_perm, "Régime permanent", zmin=0, zmax=_zmax, largeur=620, hauteur=360)
    _sonde = go.Figure()
    _sonde.add_trace(go.Scatter(x=temps_pl, y=U_pl[:, i_sonde], mode="lines", name="sous la source", line=dict(color="rgb(245,70,58)")))
    _sonde.add_trace(go.Scatter(x=temps_pl, y=U_pl.mean(axis=1), mode="lines", name="moyenne aux nœuds", line=dict(color="rgb(58,70,245)")))
    _sonde.add_hline(y=U_perm[i_sonde], line=dict(color="rgb(245,70,58)", dash="dot"), annotation_text="permanent")
    _sonde.add_vline(x=temps_pl[_k], line=dict(color="gray", width=1))
    _sonde.update_layout(title="Température au cours du temps", xaxis_title="t", yaxis_title="u", plot_bgcolor="rgba(0,0,0,0)", width=620, height=320)
    mo.vstack([_champ, _perm, _sonde])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bilan d'énergie

    En prenant $v = 1$ dans la formulation faible (c'est-à-dire en sommant toutes les équations du système), on obtient
    le bilan global
    $$
    \frac{\mathrm{d}}{\mathrm{d}t}\underbrace{\int_\Omega u}_{E(t)} = \underbrace{\int_\Omega q}_{P_{\text{entrant}}}
    - \underbrace{\mathrm{Bi}\int_{\partial\Omega} u}_{\Phi_{\text{sortant}}(t)} ,
    $$
    car $\int_\Omega \nabla u \cdot \nabla 1 = 0$. Comme $\sum_i \varphi_i = 1$, ces intégrales se calculent directement
    à partir des matrices : $E = \mathbf{1}^\top M U$, $P = \mathbf{1}^\top F$, $\Phi = \mathrm{Bi}\,\mathbf{1}^\top R\, U$.
    Avec le schéma d'Euler implicite, ce bilan est vérifié **exactement** pas à pas (aux erreurs d'arrondi près).
    """)
    return


@app.cell(hide_code=True)
def _(F_pl, M_pl, R_pl, U_pl, biot, dt_pl, go, mo, np, temps_pl):
    _E = (M_pl @ U_pl.T).sum(axis=0)
    _P = F_pl.sum()
    _Phi = biot.value * (R_pl @ U_pl.T).sum(axis=0)
    _dEdt = np.diff(_E) / dt_pl
    _residu = np.abs(_dEdt - (_P - _Phi[1:])).max()
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=temps_pl, y=np.full_like(temps_pl, _P), mode="lines", name="P entrant", line=dict(color="rgb(245,70,58)")))
    _fig.add_trace(go.Scatter(x=temps_pl, y=_Phi, mode="lines", name="Φ sortant (convection)", line=dict(color="rgb(58,70,245)")))
    _fig.add_trace(go.Scatter(x=temps_pl[1:], y=_dEdt, mode="lines", name="dE/dt (stockage)", line=dict(color="rgb(23,15,48)", dash="dash")))
    _fig.update_layout(title="Bilan d'énergie", xaxis_title="t", yaxis_title="puissance", plot_bgcolor="rgba(0,0,0,0)", width=620, height=340)
    mo.vstack([_fig, mo.md(f"Résidu maximal du bilan discret $|\\mathrm{{d}}E/\\mathrm{{d}}t - (P - \\Phi)|$ : `{_residu:.2e}` (à comparer à P = `{_P:.3f}`)")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Questions
    1. Comment évoluent la température maximale en régime permanent et le temps pour l'atteindre quand Bi augmente ?
       Interprétez physiquement les cas $\mathrm{Bi} \ll 1$ et $\mathrm{Bi} \gg 1$.
       Vers quelle condition aux limites tend-on quand $\mathrm{Bi} \to \infty$ ?
    2. Rapprochez la source d'un bord. Pourquoi la température maximale diminue-t-elle ?
    3. En régime permanent, que vaut $\Phi_{\text{sortant}}$ ? Déduisez-en, pour $\mathrm{Bi}$ petit, une estimation de la
       température moyenne $\bar u \approx P / (\mathrm{Bi} \cdot |\partial\Omega|)$ et comparez-la à la simulation.
    4. Pourquoi la matrice $K + \mathrm{Bi}\,R$ est-elle inversible alors que $K$ seule (sans condition de Dirichlet) ne l'est pas ?
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
