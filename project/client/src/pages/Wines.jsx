// client/src/pages/Wines.jsx
import React, { useEffect, useState, useCallback } from 'react';
import {
  Container,
  Typography,
  Card,
  CardMedia,
  CardContent,
  Box,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
  Autocomplete,
  TextField,
  Button,
  IconButton,
  Drawer,
  Divider
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Pagination from '@mui/material/Pagination';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getWines, getWinesCount } from '../services/wineService';

const wineTypes = ['Red','White','Sparkling','Rosé','Dessert','Fortified'];
const regionsList = ['Bordeaux','Bourgogne','Napa Valley','Piemonte','Rhone Valley','Toscana'];
const countriesList = ['Argentina','Australia','Austria','Chile','France','Germany','Italy','Portugal','Spain','United States'];
const stylesList = ['Argentinian Malbec','Californian Cabernet Sauvignon','Central Italy Red','Spanish Red','Spanish Rioja Red'];
const pairingsList = ['Aperitif','Appetizers and snacks','Beef','Blue cheese','Cured Meat','Fruity desserts','Game (deer, venison)','Goat\'s Milk Cheese','Lamb','Lean fish','Mature and hard cheese','Mild and soft cheese','Mushrooms','Pasta','Pork','Poultry','Rich fish (salmon, tuna etc)','Shellfish','Spicy food','Sweet desserts','Veal','Vegetarian'];

export default function Wines() {
  const [wines, setWines] = useState([]);
  const [limit] = useState(20);
  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Filter state
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [priceRange, setPriceRange] = useState([0, 10000]);
  const [minRating, setMinRating] = useState(null);
  const [selectedGrapes, setSelectedGrapes] = useState([]);
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [selectedCountries, setSelectedCountries] = useState([]);
  const [selectedStyles, setSelectedStyles] = useState([]);
  const [selectedPairings, setSelectedPairings] = useState([]);

  // Drawer for mobile filters
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Build filter object matching backend params
  const buildFilters = () => ({
    types: selectedTypes.join(','),
    min_price: priceRange[0],
    max_price: priceRange[1],
    min_rating: minRating,
    grapes: selectedGrapes.join(','),
    regions: selectedRegions.join(','),
    countries: selectedCountries.join(','),
    styles: selectedStyles.join(','),
    pairings: selectedPairings.join(','),
  });

  // Fetch count with optional filters
  const fetchCount = useCallback((filters = {}) => {
    getWinesCount(filters)
      .then(data => setTotalCount(data.count))
      .catch(() => setTotalCount(0));
  }, []);

  // Fetch wines with filters & page
  const fetchWines = useCallback((filters = {}, pageNum = page) => {
    setLoading(true);
    getWines({ ...filters, limit, page: pageNum })
      .then(data => setWines(Array.isArray(data) ? data : []))
      .catch(() => setWines([]))
      .finally(() => setLoading(false));
  }, [limit, page]);

  // Initial load
  useEffect(() => {
    fetchCount({});
    fetchWines({}, 1);
  }, [fetchCount, fetchWines]);

  const handleApplyFilters = () => {
    const filters = buildFilters();
    setPage(1);
    fetchCount(filters);
    fetchWines(filters, 1);
    setDrawerOpen(false);
  };

  const handlePageChange = (_, value) => {
    setPage(value);
    const filters = buildFilters();
    fetchWines(filters, value);
  };

  const totalPages = Math.ceil(totalCount / limit);

  // Filter panel JSX
  const FilterPanel = (
    <Box sx={{ width: { xs: 260, md: 300 }, p: 2 }}>
      <Typography variant="h6" gutterBottom>Filters</Typography>
      <Divider sx={{ mb:2 }} />

      {/* Wine Types */}
      <Typography variant="subtitle1">Wine Types</Typography>
      <FormGroup>
        {wineTypes.map(type => (
          <FormControlLabel
            key={type}
            control={
              <Checkbox
                checked={selectedTypes.includes(type)}
                onChange={e => {
                  setSelectedTypes(prev =>
                    e.target.checked ? [...prev, type] : prev.filter(t => t!==type)
                  );
                }}
              />
            }
            label={type}
          />
        ))}
      </FormGroup>

      {/* Price Range */}
      <Box sx={{ mt:3 }}>
        <Typography variant="subtitle1">Price Range</Typography>
        <Slider
          value={priceRange}
          onChange={(e,v)=>setPriceRange(v)}
          valueLabelDisplay="auto"
          min={0}
          max={10000}
        />
      </Box>

      {/* Min Rating */}
      <Box sx={{ mt:3 }}>
        <Typography variant="subtitle1">Min Rating</Typography>
        <Slider
          value={minRating||0}
          onChange={(e,v)=>setMinRating(v)}
          valueLabelDisplay="auto"
          min={0}
          max={5}
          step={0.1}
        />
      </Box>

      {/* Grapes */}
      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          freeSolo
          options={[]}
          value={selectedGrapes}
          onChange={(e,v)=>setSelectedGrapes(v)}
          renderInput={params => <TextField {...params} label="Grapes" />}
        />
      </Box>

      {/* Regions */}
      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={regionsList}
          value={selectedRegions}
          onChange={(e,v)=>setSelectedRegions(v)}
          renderInput={params => <TextField {...params} label="Regions" />}
        />
      </Box>

      {/* Countries */}
      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={countriesList}
          value={selectedCountries}
          onChange={(e,v)=>setSelectedCountries(v)}
          renderInput={params => <TextField {...params} label="Countries" />}
        />
      </Box>

      {/* Styles */}
      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={stylesList}
          value={selectedStyles}
          onChange={(e,v)=>setSelectedStyles(v)}
          renderInput={params => <TextField {...params} label="Styles" />}
        />
      </Box>

      {/* Pairings */}
      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={pairingsList}
          value={selectedPairings}
          onChange={(e,v)=>setSelectedPairings(v)}
          renderInput={params => <TextField {...params} label="Pairings" />}
        />
      </Box>

      <Button
        variant="contained"
        fullWidth
        sx={{ mt:4 }}
        onClick={handleApplyFilters}
      >
        Apply Filters
      </Button>
    </Box>
  );

  return (
    <Container sx={{ py:5 }}>
      {/* Mobile filter button */}
      <IconButton
        sx={{ display:{ md:'none' }, mb:2 }}
        onClick={()=>setDrawerOpen(true)}
      >
        <MenuIcon />
      </IconButton>

      <Box sx={{ display:{ xs:'block', md:'flex' }, gap:4 }}>
        {/* Sidebar on md+ */}
        <Box sx={{ display:{ xs:'none', md:'block' } }}>
          <Paper elevation={0} sx={{
            p:2,
            borderRadius:3,
            backdropFilter:'blur(10px)',
            backgroundColor:'rgba(255,255,255,0.05)',
            border:'1px solid rgba(255,255,255,0.15)'
          }}>
            {FilterPanel}
          </Paper>
        </Box>

        {/* Drawer for mobile */}
        <Drawer
          open={drawerOpen}
          onClose={()=>setDrawerOpen(false)}
        >
          {FilterPanel}
        </Drawer>

        {/* Wine grid */}
        <Box sx={{ flex:1 }}>
          <Typography variant="h4" align="center" gutterBottom sx={{
            textTransform:'uppercase',
            color:'text.primary'
          }}>
            Explore Wines
          </Typography>

          {loading ? (
            <Typography align="center">Loading…</Typography>
          ) : wines.length === 0 ? (
            <Typography align="center">No wines found.</Typography>
          ) : (
            <Box sx={{
              display:'grid',
              gridTemplateColumns:'repeat(auto-fit, minmax(220px,1fr))',
              gap:2
            }}>
              {wines.map(wine=>(
                <motion.div key={wine.id} whileHover={{scale:1.05}} transition={{duration:0.3}}>
                  <Card sx={{cursor:'pointer'}} onClick={()=>navigate(`/wine-details/${wine.id}`)}>
                    <CardMedia
                      component="img"
                      sx={{height:300, objectFit:'contain'}}
                      image={wine.image_url||'https://source.unsplash.com/random/400x300/?wine'}
                      alt={wine.wine_name}
                    />
                    <CardContent>
                      <Typography variant="h6" noWrap sx={{
                        textTransform:'uppercase',
                        color:'text.primary'
                      }}>
                        {wine.wine_name}
                      </Typography>
                      <Typography variant="body2" noWrap sx={{
                        textTransform:'uppercase',
                        color:'text.secondary'
                      }}>
                        ${wine.price} • {wine.rating} Stars
                      </Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </Box>
          )}

          {!loading && totalPages > 1 && (
            <Box sx={{display:'flex', justifyContent:'center', mt:4}}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
              />
            </Box>
          )}
        </Box>
      </Box>
    </Container>
  );
}
