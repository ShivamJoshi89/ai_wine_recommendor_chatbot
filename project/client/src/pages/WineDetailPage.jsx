// client/src/pages/WineDetailPage.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Button,
  Grid,
  CardMedia,
  Box,
  Divider
} from '@mui/material';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import api from '../services/api';
import ReviewSection from '../components/reviews/ReviewSection';

const WineDetailPage = () => {
  const { id } = useParams();
  const [wine, setWine] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get(`/wines/${id}`)
      .then((response) => {
        setWine(response.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching wine details:', err);
        setError('Error fetching wine details');
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <Container sx={{ py: 5 }}>
        <Typography variant="h6" sx={{ color: 'text.primary' }}>
          Loading...
        </Typography>
      </Container>
    );
  }

  if (error || !wine) {
    return (
      <Container sx={{ py: 5 }}>
        <Typography variant="h6" color="error">
          {error || 'Wine not found'}
        </Typography>
      </Container>
    );
  }

  return (
    <Container
      sx={{
        py: 5,
        position: 'relative',
        zIndex: 2,
        mt: 4,
        // Remove semi-transparent background, borderRadius, and boxShadow
      }}
    >
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
        {/* 
          The Paper is set to transparent with no shadow, 
          so the background remains dark from your theme.
        */}
        <Paper elevation={3} sx={{ p: 4, backgroundColor: 'transparent', boxShadow: 'none' }}>
          <Grid container spacing={4}>
            {/* Image Section */}
            <Grid item xs={12} md={6}>
              <Box sx={{ borderRadius: 2, overflow: 'hidden', boxShadow: 3 }}>
                <CardMedia
                  component="img"
                  height="400"
                  image={wine.image_url || 'https://source.unsplash.com/random/400x300/?wine'}
                  alt={wine.wine_name}
                  sx={{ objectFit: 'cover' }}
                />
              </Box>
            </Grid>

            {/* Details Section */}
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography
                  variant="h4"
                  gutterBottom
                  sx={{ textTransform: 'uppercase', color: 'text.primary' }}
                >
                  {wine.wine_name}
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ color: 'text.secondary', textTransform: 'uppercase' }}
                >
                  {wine.winery} • {wine.country} • {wine.region}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography
                    variant="h6"
                    sx={{ color: 'primary.main', textTransform: 'uppercase' }}
                  >
                    ${wine.price}
                  </Typography>
                  <Typography
                    variant="body1"
                    sx={{ color: 'text.secondary', textTransform: 'uppercase' }}
                  >
                    {wine.rating} Stars
                  </Typography>
                </Box>
                <Divider sx={{ my: 2 }} />
                <Typography
                  variant="subtitle1"
                  fontWeight="bold"
                  sx={{ textTransform: 'uppercase', color: 'text.primary' }}
                >
                  Description
                </Typography>
                <Typography variant="body1" sx={{ color: 'text.primary' }}>
                  {wine.wine_description_1}
                </Typography>
                {wine.wine_description_2 && (
                  <Typography variant="body1" sx={{ color: 'text.primary' }}>
                    {wine.wine_description_2}
                  </Typography>
                )}
                <Divider sx={{ my: 2 }} />
                <Typography
                  variant="subtitle1"
                  fontWeight="bold"
                  sx={{ textTransform: 'uppercase', color: 'text.primary' }}
                >
                  Food Pairing:
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {wine.food_pairing}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography
                  variant="subtitle1"
                  fontWeight="bold"
                  sx={{ textTransform: 'uppercase', color: 'text.primary' }}
                >
                  Additional Details:
                </Typography>
                <Typography variant="body2" sx={{ textTransform: 'uppercase', color: 'text.secondary' }}>
                  Grape Type: {wine.grape_type_list}
                </Typography>
                <Typography variant="body2" sx={{ textTransform: 'uppercase', color: 'text.secondary' }}>
                  Alcohol Content: {wine.alcohol_content}%
                </Typography>
                <Typography variant="body2" sx={{ textTransform: 'uppercase', color: 'text.secondary' }}>
                  Bottle Closure: {wine.bottle_closure}
                </Typography>
                <Box sx={{ mt: 3 }}>
                  <Button variant="contained" size="large" sx={{ textTransform: 'uppercase' }}>
                    Add to Cart
                  </Button>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </motion.div>

      {/* Divider before review section */}
      <Divider sx={{ my: 4 }} />

      {/* Review Section integration */}
      <ReviewSection wineId={wine.id} token={localStorage.getItem('token')} />
    </Container>
  );
};

export default WineDetailPage;
