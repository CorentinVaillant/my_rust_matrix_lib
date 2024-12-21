#[derive(Debug)]
pub enum MatrixError{
    SizeNotMatch,
    Other(String),
}

impl Into<String> for MatrixError{
    fn into(self) -> String {
        match self {
            Self::SizeNotMatch => "The sizes does not matches".to_string(),
            Self::Other(s) =>format!("Other error :{}",s)
        }
    }
}

impl core::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
        "{}", self.to_string()
        
        )
    }
}

impl core::error::Error for MatrixError {}