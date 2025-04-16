// client/src/pages/Home.jsx
import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardMedia
} from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getFeaturedWine } from '../services/wineService';

const Home = () => {
  const [featuredWines, setFeaturedWines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    setLoading(true);
    getFeaturedWine()
      .then((data) => {
        setFeaturedWines(data);
        setLoading(false);
      })
      .catch(() => {
        setError('Error fetching featured wines. Please try again later.');
        setLoading(false);
      });
  }, []);

  const handleExploreClick = () => {
    navigate('/wines');
  };

  const handleWineClick = (wine) => {
    navigate(`/wine-details/${wine.id}`);
  };

  return (
    <>
      {/* HERO SECTION - Transparent container with absolutely positioned text */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 2,
          height: '300px', // Define the area for the hero text
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: '150px', // Increased from '50px' to push the animation down
            left: '50%',
            transform: 'translateX(-50%)',
            textAlign: 'center',
          }}
        >
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
            <Typography variant="h2" sx={{ textTransform: 'uppercase', color: 'text.primary', mb: 2 }}>
              Welcome to Vino-Sage
            </Typography>
            <Typography variant="h6" sx={{ color: 'text.secondary', mb: 2 }}>
              Discover your perfect wine with personalized recommendations.
            </Typography>
            <Button variant="contained" size="large" sx={{ textTransform: 'uppercase' }} onClick={handleExploreClick}>
              Explore Wines
            </Button>
          </motion.div>
        </Box>
      </Box>

      {/* BEST PICKS SECTION - Appears after scrolling, transparent background */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 2,
          mt: 70, // Pushes section further down the page
          px: { xs: 2, md: 4 },
        }}
      >
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
          <Typography variant="h4" align="center" gutterBottom sx={{ textTransform: 'uppercase', color: 'text.primary' }}>
            Best Picks
          </Typography>
          <Typography variant="body1" align="center" sx={{ mt: 1, mb: 3, color: 'text.primary' }}>
            Get great value and seamless service with these brilliant wines, available direct from Vino-Sage and a selection of our best merchant partners
          </Typography>

          {loading ? (
            <Typography variant="h6" align="center" sx={{ color: 'text.primary' }}>
              Loading featured wines...
            </Typography>
          ) : error ? (
            <Typography variant="h6" align="center" color="error">
              {error}
            </Typography>
          ) : featuredWines && featuredWines.length > 0 ? (
            <Box sx={{ mt: 2, overflowX: 'auto', whiteSpace: 'nowrap' }}>
              <Box sx={{ display: 'inline-flex', gap: 2, pb: 2 }}>
                {featuredWines.map((wine, index) => (
                  <motion.div
                    key={index}
                    whileHover={{ scale: 1.05 }}
                    transition={{ duration: 0.3 }}
                    style={{ display: 'inline-block' }}
                  >
                    <Card sx={{ width: 220, cursor: 'pointer' }} onClick={() => handleWineClick(wine)}>
                      <CardMedia
                        component="img"
                        sx={{
                          height: 300,
                          objectFit: 'contain',
                        }}
                        image={
                          wine.image_url ||
                          'https://source.unsplash.com/random/400x300/?wine'
                        }
                        alt={wine.wine_name}
                      />
                      <CardContent>
                        <Typography variant="h6" noWrap sx={{ textTransform: 'uppercase', color: 'text.primary' }}>
                          {wine.wine_name}
                        </Typography>
                        <Typography variant="body2" noWrap sx={{ textTransform: 'uppercase', color: 'text.secondary' }}>
                          ${wine.price} • {wine.rating} Stars
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </Box>
            </Box>
          ) : (
            <Typography variant="h6" align="center" sx={{ color: 'text.primary' }}>
              No featured wines available.
            </Typography>
          )}
        </motion.div>
      </Box>
    </>
  );
};

export default Home;
