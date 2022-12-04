use num::{Float, Complex, One, Zero};
use array_matrix::*;
use transfer_function::TfS;

/// State-space equation matrices A, B, C and D contained in one structure
#[derive(Clone, Copy)]
pub struct StateSpace<F: Float, const VARCOUNT: usize>
where
    [[F; VARCOUNT]; VARCOUNT]: SquareMatrix,
    [[F; 1]; VARCOUNT]: Matrix,
    [[F; VARCOUNT]; 1]: Matrix
{
    pub a: [[F; VARCOUNT]; VARCOUNT],
    pub b: [[F; 1]; VARCOUNT],
    pub c: [[F; VARCOUNT]; 1],
    pub d: F
}

impl<F: Float, const VARCOUNT: usize> StateSpace<F, VARCOUNT>
where
    [[F; VARCOUNT]; VARCOUNT]: SquareMatrix,
    [[F; 1]; VARCOUNT]: Matrix,
    [[F; VARCOUNT]; 1]: Matrix
{
    /// Creates a new state-space struct in control-canonical form
    /// 
    /// # Arguments
    /// 
    /// * `a` - Transfer function denominator polynomial
    /// * `b` - Transfer function enumerator polynomial
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let a = [1.0, 2.0, 3.0];
    /// let b = [4.0, 5.0, 6.0];
    /// 
    /// let ss = StateSpace::new_control_canonical_form(a, b);
    /// 
    /// assert_eq!(ss.a, [
    ///     [-a[0], -a[1], -a[2]],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0]
    /// ]);
    /// assert_eq!(ss.b, [
    ///     [1.0],
    ///     [0.0],
    ///     [0.0]
    /// ]);
    /// assert_eq!(ss.c, [
    ///     [b[0], b[1], b[2]]
    /// ]);
    /// assert_eq!(ss.d, 0.0);
    /// ```
    pub fn new_control_canonical_form(a: [F; VARCOUNT], b: [F; VARCOUNT]) -> Self
    {
        Self {
            a: array_init::array_init(|r| if r == 0 {
                a.map(|ai| -ai)
            } else {
                array_init::array_init(|c| if r == c + 1 {F::one()} else {F::zero()})
            }),
            b: array_init::array_init(|r| if r == 0 {[F::one()]} else {[F::zero()]}),
            c: [b],
            d: F::zero()
        }
    }

    /// Creates a new state-space struct in observer-canonical form
    /// 
    /// # Arguments
    /// 
    /// * `a` - Transfer function denominator polynomial
    /// * `b` - Transfer function enumerator polynomial
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let a = [1.0, 2.0, 3.0];
    /// let b = [4.0, 5.0, 6.0];
    /// 
    /// let ss = StateSpace::new_observer_canonical_form(a, b);
    /// 
    /// assert_eq!(ss.a, [
    ///     [-a[0], 1.0, 0.0],
    ///     [-a[1], 0.0, 1.0],
    ///     [-a[2], 0.0, 0.0]
    /// ]);
    /// assert_eq!(ss.b, [
    ///     [b[0]],
    ///     [b[1]],
    ///     [b[2]]
    /// ]);
    /// assert_eq!(ss.c, [
    ///     [1.0, 0.0, 0.0]
    /// ]);
    /// assert_eq!(ss.d, 0.0);
    /// ```
    pub fn new_observer_canonical_form(a: [F; VARCOUNT], b: [F; VARCOUNT]) -> Self
    {
        Self {
            a: array_init::array_init(|r| if r == 0 {
                a.map(|ai| -ai)
            } else {
                array_init::array_init(|c| if r == c + 1 {F::one()} else {F::zero()})
            }).transpose(),
            b: [b].transpose(),
            c: [array_init::array_init(|r| if r == 0 {F::one()} else {F::zero()})],
            d: F::zero()
        }
    }

    /// Returns the controllability-matrix of the state-space equation
    /// 
    /// 𝒞
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let cc = ss.controllability()
    /// ```
    pub fn controllability(&self) -> [[F; VARCOUNT]; VARCOUNT]
    {
        array_init::array_init(|n| {
            let mut b: [[F; 1]; VARCOUNT] = self.b;
            for _ in 0..n
            {
                b = self.a.mul(b);
            }
            b.transpose()[0]
        }).transpose()
    }

    /// Checks if the controllability-matrix is nonzero
    /// 
    /// det(𝒞) != 0
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// ss.is_controllable()
    /// ```
    pub fn is_controllable(&self) -> bool
    where
        [[F; VARCOUNT]; VARCOUNT]: Det<Output = F>
    {
        !self.controllability().det().is_zero()
    }
    
    /// Returns the observability-matrix of the state-space equation
    /// 
    /// 𝒪
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let cc = ss.observability()
    /// ```
    pub fn observability(&self) -> [[F; VARCOUNT]; VARCOUNT]
    {
        array_init::array_init(|n| {
            let mut c: [[F; VARCOUNT]; 1] = self.c;
            for _ in 0..n
            {
                c = c.mul(self.a);
            }
            c[0]
        })
    }

    /// Checks if the observability-matrix is nonzero
    /// 
    /// det(𝒪) != 0
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// ss.is_observable()
    /// ```
    pub fn is_observable(&self) -> bool
    where
        [[F; VARCOUNT]; VARCOUNT]: Det<Output = F>
    {
        !self.observability().det().is_zero()
    }

    /// Transforms the form of the state-space system by a transformation matrix
    /// 
    /// A' = T⁻¹AT
    /// 
    /// B' = T⁻¹B
    /// 
    /// C' = CT
    /// 
    /// D' = D
    /// 
    /// # Arguments
    /// 
    /// * `t` - The transformation matrix
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let sst = ss.transform_state(t)
    /// ```
    pub fn transform_state(&self, t: [[F; VARCOUNT]; VARCOUNT]) -> Option<Self>
    where [[F; VARCOUNT]; VARCOUNT]: MInv<Output = [[F; VARCOUNT]; VARCOUNT]>
    {
        t.inv().map(|t_| Self {
            a: t_.mul(self.a).mul(t),
            b: t_.mul(self.b),
            c: self.c.mul(t),
            d: self.d
        })
    }

    /// Transforms any state-space system into control-canonical form
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let ssc = ss.transform_into_control_canonical_form()
    /// ```
    pub fn transform_into_control_canonical_form(&self) -> Option<Self>
    where [[F; VARCOUNT]; VARCOUNT]: MInv<Output = [[F; VARCOUNT]; VARCOUNT]>
    {
        self.controllability().inv().map(|cc_| {
            let tn = [array_init::array_init(|c| if c == 0 {F::one()} else {F::zero()})].mul(cc_);
            let mut tna = tn;
            let mut t = array_init::array_init(|_| {
                let tn = tna;
                tna = tna.mul(self.a);
                tn[0]
            });
            t.reverse();
            return self.transform_state(t)
        }).flatten()
    }

    /// Transforms any state-space system into observer-canonical form
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let sso = ss.transform_into_observer_canonical_form()
    /// ```
    pub fn transform_into_observer_canonical_form(&self) -> Option<Self>
    where [[F; VARCOUNT]; VARCOUNT]: MInv<Output = [[F; VARCOUNT]; VARCOUNT]>
    {
        self.transform_into_control_canonical_form().map(|ccf| Self {
            a: ccf.a.transpose(),
            b: ccf.c.transpose(),
            c: ccf.b.transpose(),
            d: ccf.d
        })
    }

    /// Returns the derivative of the internal variables according to the state-space equations
    /// 
    /// # Arguments
    /// 
    /// * `x` - The state of the internal state-space variables organized in a collumn-vector
    /// * `u` - The system input state
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let dt = 0.0001;
    /// let mut x = [
    ///     [0.0],
    ///     [0.0]
    /// ];
    /// let mut u = 0.0;
    /// for _ in 0..10000
    /// {
    ///     x = x.add(ss.dxdt(x, u).mul(dt))
    ///     u = 1.0;
    /// }
    /// let y_ss = ss.y(x, u);
    /// // Steady state error:
    /// let e_ss = y_ss - u;
    /// ```
    pub fn dxdt(&self, x: [[F; 1]; VARCOUNT], u: F) -> [[F; 1]; VARCOUNT]
    {
        self.a.mul(x).add(self.b.mul(u))
    }

    /// Returns the output state according to the state-space equations
    /// 
    /// # Arguments
    /// 
    /// * `x` - The state of the internal state-space variables organized in a collumn-vector
    /// * `u` - The system input state
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let dt = 0.0001;
    /// let mut x = [[0.0]; 2];
    /// let mut u = 0.0;
    /// for _ in 0..10000
    /// {
    ///     x = x.add(ss.dxdt(x, u).mul(dt));
    ///     u = 1.0;
    /// }
    /// let y_ss = ss.y(x, u);
    /// // Steady state error:
    /// let e_ss = y_ss - u;
    /// ```
    pub fn y(&self, x: [[F; 1]; VARCOUNT], u: F) -> F
    {
        self.c.mul(x)[0][0] + self.d*u
    }

    /// Returns the transfer function of the system
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let tf = ss.tf();
    /// ```
    pub fn tf(&self) -> TfS<F>
    where
        [[TfS<F>; VARCOUNT]; VARCOUNT]: SquareMatrix
            + Adj<Output = [[TfS<F>; VARCOUNT]; VARCOUNT]>
            + Det<Output = TfS<F>>,
        [[TfS<F>; 1]; VARCOUNT]: Matrix,
        [[TfS<F>; VARCOUNT]; 1]: Matrix,
        [[TfS<F>; 1]; 1]: SquareMatrix,
    {
        let is = matrix_init(|r, c| TfS::S*if r == c {Complex::one()} else {Complex::zero()});
        let is_sub_a = is.sub(self.a);
        let is_sub_a_inv = is_sub_a.adj().mul(is_sub_a.det().inv());
        let c = self.c.map(|ci| ci.map(|cij| TfS::from(cij)));
        let b = self.b.map(|bi| bi.map(|bij| TfS::from(bij)));
        c.mul(is_sub_a_inv.mul(b))[0][0].clone() + self.d
    }
}

impl<F: Float, const VARCOUNT: usize> Into<TfS<F>>
for
    StateSpace<F, VARCOUNT>
where
    [[TfS<F>; VARCOUNT]; VARCOUNT]: SquareMatrix
        + Adj<Output = [[TfS<F>; VARCOUNT]; VARCOUNT]>
        + Det<Output = TfS<F>>,
    [[TfS<F>; 1]; VARCOUNT]: Matrix,
    [[TfS<F>; VARCOUNT]; 1]: Matrix,
    [[TfS<F>; 1]; 1]: SquareMatrix,
    [[F; VARCOUNT]; VARCOUNT]: SquareMatrix,
    [[F; 1]; VARCOUNT]: Matrix,
    [[F; VARCOUNT]; 1]: Matrix
{
    fn into(self) -> TfS<F>
    {
        self.tf()
    }
}