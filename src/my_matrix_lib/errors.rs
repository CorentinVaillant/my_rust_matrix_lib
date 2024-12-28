#[derive(Debug)]
pub enum MatrixError{
    SizeNotMatch(usize,usize),
    WidhtNotMatch,
    HeigthNotMach,
    Other(String),
}

impl Into<String> for MatrixError{
    fn into(self) -> String {
        match &self {
            Self::SizeNotMatch(size1,size2) => format!("The sizes does not matches ({} != {})",size1,size2),
            Self::HeigthNotMach=> "The heights does not matches".to_string(),
            Self::WidhtNotMatch=> "The widths does not matches".to_string(),
            
            Self::Other(s) =>
                format!("Other error :{}",s)
        }
        
    }
}
impl Into<String> for &MatrixError{
    fn into(self) -> String {
        match self {
            MatrixError::SizeNotMatch(size1,size2) => format!("The sizes does not matches ({} != {})",size1,size2),
            MatrixError::HeigthNotMach=> "The heights does not matches".to_string(),
            MatrixError::WidhtNotMatch=> "The widths does not matches".to_string(),

            MatrixError::Other(s) =>
                format!("Other error :{}",s)
        }
    }
}

impl core::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
        "{}", <&Self as Into<String>>::into(self)
        
        )
    }
}

impl core::error::Error for MatrixError {}