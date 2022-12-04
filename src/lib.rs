pub mod state_space;

pub use state_space::*;

#[cfg(test)]
mod tests {
    use array_matrix::{MAdd, MMul};

    use crate::StateSpace;

    #[test]
    fn step()
    {
        let ss = StateSpace::new_control_canonical_form([10.0, 2.0], [0.0, 1.0]);

        let mut x = [[0.0]; 2];
        let mut y = vec![];
        let mut u = 0.0;

        const FS: f64 = 100.0;
        const T: f64 = 20.0;
        const N: usize = (T*FS) as usize;
        let dt = 1.0/FS;

        for _ in 0..N
        {
            y.push(ss.y(x, u));
            x = x.add(ss.dxdt(x, u).mul(dt));
            u = 1.0;
        }

        let y_str: Vec<String> = y.iter().map(|yn| yn.to_string()).collect();

        println!("{}", y_str.join(", "));
    }

    #[test]
    fn steady_state_error()
    {
        let ss = StateSpace::new_control_canonical_form([10.0, 1.0], [0.0, 1.0]);
        
        let mut x = [[0.0]; 2];
        let mut u = 0.0;

        const FS: f64 = 100.0;
        const T: f64 = 200.0;
        const N: usize = (T*FS) as usize;
        let dt = 1.0/FS;

        for _ in 0..N
        {
            x = x.add(ss.dxdt(x, u).mul(dt));
            u = 1.0;
        }
        let y_ss = ss.y(x, u);
        // Steady state error:
        let e_ss = y_ss - u;
        println!("{}", e_ss)
    }
}
