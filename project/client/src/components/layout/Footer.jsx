// client/src/components/layout/Footer.jsx
import React from 'react';
import { Box, Typography, Container, Link, IconButton } from '@mui/material';
import { Facebook, Twitter, Instagram } from '@mui/icons-material';

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        backgroundColor: '#44334A', // Dark background that blends with your palette
        color: '#ffffff',           // White text for contrast
        py: 3,
        mt: 4,
      }}
    >
      <Container maxWidth="lg">
        {/* Top Row */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            mb: 2,
          }}
        >
          <Typography variant="body1">
            © {new Date().getFullYear()} Wine Recommender. All rights reserved.
          </Typography>
          <Box>
            <Link href="#" color="inherit" sx={{ mx: 1 }}>
              Terms of Service
            </Link>
            <Link href="#" color="inherit" sx={{ mx: 1 }}>
              Privacy Policy
            </Link>
            <Link href="#" color="inherit" sx={{ mx: 1 }}>
              Contact Us
            </Link>
          </Box>
        </Box>

        {/* Bottom Row: Social Media Icons */}
        <Box sx={{ textAlign: 'center' }}>
          <IconButton href="#" aria-label="facebook" sx={{ color: 'inherit' }}>
            <Facebook />
          </IconButton>
          <IconButton href="#" aria-label="twitter" sx={{ color: 'inherit' }}>
            <Twitter />
          </IconButton>
          <IconButton href="#" aria-label="instagram" sx={{ color: 'inherit' }}>
            <Instagram />
          </IconButton>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
