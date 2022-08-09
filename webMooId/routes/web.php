<?php

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', 'BerandaController@welcome')->name('welcome');

Route::get('/beranda', 'BerandaController@index')->name('beranda');

Route::resource('/tempat', 'TempatController');

Route::resource('/sapi', 'SapiController');
Route::get('/sapi_aktivitas', 'SapiController@aktivitas')->name('sapi.aktivitas');

Route::get('/obat', 'ObatController@index')->name('obat.index');
Route::get('/obat/show/{id}', 'ObatController@show')->name('obat.show');
Route::get('/obat/cart', 'ObatController@cart')->name('obat.cart');
Route::get('/obat/track', 'ObatController@track')->name('obat.track');
Route::get('/obat/toko_obat/{id}', 'ObatController@show_toko_obat')->name('obat.show_toko_obat');

Route::resource('/petugas', 'PetugasController');

Route::resource('/dokter', 'DokterController');

Route::resource('/artikel', 'ArtikelController');

Route::get('/history', 'HistoryController@index')->name('history.index');
Route::get('/history/create', 'HistoryController@create')->name('history.create');

Route::get('/profile', 'ProfileController@index')->name('profile');
Route::get('/profile/edit', 'ProfileController@edit')->name('profile.edit');
Route::patch('/profile/update', 'ProfileController@update')->name('profile.update');

Route::get('/login', 'AuthController@login');

Route::get('/register', 'AuthController@register');
