import React from 'react';
import Header from './Header';
import Footer from './Footer';
import { Container, Box } from '@mui/material';
import { useLocation } from 'react-router-dom';
import BackgroundFlowMulti from '../common/BackgroundFlowMulti';

const Layout = ({ children }) => {
  const location = useLocation();
  // Define the routes where you want to hide the background animation.
  const hideBackground =
    location.pathname.startsWith('/wines') ||
    location.pathname.startsWith('/wine-details');

  return (
    <Box sx={{ position: 'relative', minHeight: '100vh', overflow: 'hidden' }}>
      {/* Render the background animation only if not on the hidden routes */}
      {!hideBackground && <BackgroundFlowMulti />}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1, // ensures content is above the background
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
        }}
      >
        <Header />
        <Container sx={{ flex: 1, mt: 2 }}>{children}</Container>
        <Footer />
      </Box>
    </Box>
  );
};

export default Layout;
