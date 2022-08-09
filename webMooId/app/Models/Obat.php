<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Obat extends Model
{
    // Table Name
    protected $table = 'obat';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_toko_obat',
        'tipe_obat',
        'nama_obat',
        'harga_obat',
        'deskripsi',
        'avatar',
    ];

    public function toko_obat()
    {
        return $this->belongsTo('App\Models\Toko_Obat','id_toko_obat');
    }

    public function stok_obat()
    {
        return $this->hasMany('App\Models\Stok_Obat','id_obat');
    }

    public function rekam_medis()
    {
        return $this->hasMany('App\Models\Rekam_Medis','id_obat');
    }

    public function transaksi()
    {
        return $this->hasMany('App\Models\Transaksi','id_obat');
    }
}
