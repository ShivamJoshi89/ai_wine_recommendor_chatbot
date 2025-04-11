// client/src/theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark', // using dark mode so the dark background is applied
    primary: {
      main: '#8D77A8',      // Primary actions, buttons, etc.
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#C4ADDD',      // Secondary accents
      contrastText: '#ffffff',
    },
    background: {
      default: '#1B0E20',   // Overall background
      paper: '#44334A',     // Surfaces like cards, dialogs, etc.
    },
    text: {
      primary: '#D1C0EC',   // Main text color
      secondary: '#C4ADDD', // Secondary text color
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h1: {
      color: '#D1C0EC',
      fontWeight: 700,
      fontSize: '2.5rem',
      marginBottom: '1rem',
    },
    h2: {
      color: '#D1C0EC',
      fontWeight: 700,
      fontSize: '2rem',
      marginBottom: '0.75rem',
    },
    h3: {
      color: '#D1C0EC',
      fontWeight: 600,
      fontSize: '1.75rem',
      marginBottom: '0.75rem',
    },
    h4: {
      color: '#D1C0EC',
      fontWeight: 600,
      fontSize: '1.5rem',
      marginBottom: '0.5rem',
    },
    h5: {
      color: '#D1C0EC',
      fontWeight: 500,
      fontSize: '1.25rem',
      marginBottom: '0.5rem',
    },
    h6: {
      color: '#C4ADDD',
      fontWeight: 500,
      fontSize: '1rem',
      marginBottom: '0.5rem',
    },
    body1: {
      color: '#D1C0EC',
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      color: '#C4ADDD',
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          padding: '8px 16px',
          fontWeight: 500,
          transition: 'background-color 0.3s ease',
          '&:hover': {
            // Slightly darken the primary color on hover
            backgroundColor: '#775a8c',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0px 2px 4px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          paddingLeft: '16px',
          paddingRight: '16px',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '16px',
          boxShadow: '0px 4px 8px rgba(0,0,0,0.05)',
        },
      },
    },
  },
});

export default theme;
